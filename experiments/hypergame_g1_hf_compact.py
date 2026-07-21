import os,sys,math,time,copy,json,csv,random,hashlib,argparse,shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

torch.set_num_threads(max(1,min(4,os.cpu_count() or 1)))
try: torch.set_num_interop_threads(1)
except: pass
A=(-1,0,1); BOS=3; GAMES=('ring','pong','mountain','keydoor')

def seed(x):
 random.seed(x); np.random.seed(x); torch.manual_seed(x)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(x)
def ai(a): return a+1
def av(i): return i-1
def ring_delta(a,b,n):
 d=(b-a)%n
 return d-n if d>n/2 else d
def refl(x,m):
 u=x%(2*m); return u if u<=m else 2*m-u

def pars(g,ood=False):
 if g=='ring': return dict(n=47 if ood else 31,t0=6 if ood else 4,t1=10 if ood else 8,max=10 if ood else 8)
 if g=='pong': return dict(w=11 if ood else 9,h=17 if ood else 13,sp=3 if ood else 2,max=11 if ood else 9)
 if g=='mountain': return dict(force=.00105 if ood else .0012,grav=.0027 if ood else .0025,max=110 if ood else 90)
 return dict(n=23 if ood else 17,max=(69 if ood else 51))
def reset(g,r,ood=False):
 p=pars(g,ood)
 if g=='ring':
  t=int(r.integers(p['t0'],p['t1']+1)); v=int(r.choice([-2,-1,0,1,2])); hit=int(r.integers(p['n'])); off=int(r.integers(-t,t+1))
  return np.array([(hit-off)%p['n'],(hit-v*t)%p['n'],v,t,0,0],float)
 if g=='pong':
  bx=int(r.integers(max(3,p['w']//2),p['w'])); by=int(r.integers(p['h'])); vy=int(r.choice([x for x in range(-p['sp'],p['sp']+1) if x])); hit=refl(by+vy*bx,p['h']-1); pad=int(np.clip(hit+r.integers(-bx,bx+1),0,p['h']-1))
  return np.array([pad,by,vy,bx,0,0],float)
 if g=='mountain': return np.array([r.uniform(-.62,-.42),0,0,0,0,0],float)
 n=p['n']; pos=int(r.integers(n//3,2*n//3+1))
 if r.random()<.5: key=int(r.integers(0,max(1,pos))); door=int(r.integers(min(n-1,pos+1),n))
 else: door=int(r.integers(0,max(1,pos))); key=int(r.integers(min(n-1,pos+1),n))
 return np.array([pos,key,door,0,0,0],float)
def oracle(g,s,ood=False):
 p=pars(g,ood)
 if g=='ring': return int(np.sign(ring_delta(s[0],(s[1]+s[2]*s[3])%p['n'],p['n'])))
 if g=='pong': return int(np.sign(refl(s[1]+s[2]*s[3],p['h']-1)-s[0]))
 if g=='mountain': return -1 if s[1]<=0 else 1
 target=s[2] if s[3] else s[1]; return int(np.sign(target-s[0]))
def step(g,s,a,ood=False):
 p=pars(g,ood); a=int(np.clip(a,-1,1)); x=s.copy()
 if g=='ring':
  x[0]=(x[0]+a)%p['n']; x[1]=(x[1]+x[2])%p['n']; x[3]-=1; x[4]+=1; done=x[3]<=0; suc=done and x[0]==x[1]; return x,(1 if suc else -1) if done else -.01,done,suc
 if g=='pong':
  x[0]=np.clip(x[0]+a,0,p['h']-1); y=x[1]+x[2]; vy=x[2]; m=p['h']-1
  while y<0 or y>m:
   if y<0: y=-y; vy=-vy
   if y>m: y=2*m-y; vy=-vy
  x[1]=y; x[2]=vy; x[3]-=1; x[4]+=1; done=x[3]<=0; suc=done and abs(x[0]-x[1])<=1; return x,(1 if suc else -1) if done else -.01,done,suc
 if g=='mountain':
  v=np.clip(x[1]+p['force']*a-p['grav']*math.cos(3*x[0]),-.07,.07); pos=np.clip(x[0]+v,-1.2,.6)
  if pos<=-1.2 and v<0: v=0
  x[0]=pos; x[1]=v; x[2]+=1; done=pos>=.5 or x[2]>=p['max']; suc=pos>=.5; return x,(1 if suc else -1) if done else -.01,done,suc
 x[0]=np.clip(x[0]+a,0,p['n']-1); x[3]=1 if x[3] or x[0]==x[1] else 0; x[4]+=1; suc=bool(x[3] and x[0]==x[2]); done=suc or x[4]>=p['max']; return x,(1 if suc else -1) if done else -.01,done,suc
def progress(g,s,ood=False):
 p=pars(g,ood)
 if g=='ring': return 1-abs(ring_delta(s[0],(s[1]+s[2]*max(0,s[3]))%p['n'],p['n']))/(p['n']/2)
 if g=='pong': return 1-abs(refl(s[1]+s[2]*s[3],p['h']-1)-s[0])/(p['h']-1)
 if g=='mountain': return .65*(s[0]+1.2)/1.7+.35*(abs(s[1])/.07)
 target=s[2] if s[3] else s[1]; return (.5 if s[3] else 0)+.5*(1-abs(s[0]-target)/(p['n']-1))
def raw(g,s,ood=False):
 p=pars(g,ood)
 if g=='ring': return np.array([2*s[0]/(p['n']-1)-1,2*s[1]/(p['n']-1)-1,s[2]/2,2*s[3]/p['max']-1,0,0,0,0],np.float32)
 if g=='pong': return np.array([2*s[0]/(p['h']-1)-1,2*s[1]/(p['h']-1)-1,s[2]/p['sp'],2*s[3]/(p['w']-1)-1,0,0,2*s[0]/(p['h']-1)-1,1-2*s[0]/(p['h']-1)],np.float32)
 if g=='mountain':
  q=2*(s[0]+1.2)/1.8-1; return np.array([q,1,s[1]/.07,1-2*s[2]/p['max'],math.cos(3*s[0]),0,q,-q],np.float32)
 q=lambda z:2*z/(p['n']-1)-1
 return np.array([q(s[0]),q(s[1]),0,1-2*s[4]/p['max'],q(s[2]),s[3],q(s[0]),-q(s[0])],np.float32)
def hand(g,s,ood=False):
 p=pars(g,ood); r=raw(g,s,ood)
 if g=='ring': c=[s[0]/p['n'],s[1]/p['n'],(s[2]+2)/4,s[3]/p['max']]; topo=[1,0,0,0,0,1,1,1]; fac=2*math.pi
 elif g=='pong': c=[s[0]/(p['h']-1),s[1]/(p['h']-1),s[3]/(p['w']-1),(s[1] if s[2]>=0 else 2*(p['h']-1)-s[1])/(2*(p['h']-1))]; topo=[0,1,0,0,1,1,1,1]; fac=2*math.pi
 elif g=='mountain': c=[(s[0]+1.2)/1.8,.5*(s[1]/.07+1),.5*(math.cos(3*s[0])+1),s[2]/p['max']]; topo=[0,0,1,0,1,0,1,0]; fac=math.pi
 else: c=[s[0]/(p['n']-1),s[1]/(p['n']-1),s[2]/(p['n']-1),s[3]]; topo=[0,0,0,1,1,0,1,1]; fac=math.pi
 sp=[]
 for z in c:
  for k in (1,2): sp += [math.sin(fac*k*z),math.cos(fac*k*z)]
 ds=[c[1]-c[0],c[2]-c[0],c[2]-c[1],c[3]-c[0]]; pr=[]
 for z in ds:
  for k in (1,2): pr += [math.sin(fac*k*z),math.cos(fac*k*z)]
 return np.r_[r,np.array(sp,np.float32),np.array(pr,np.float32),np.array(topo,np.float32)].astype(np.float32)

def collect(g,n,r,ctx,explore=.18,ood=False):
 out=[]
 for _ in range(n):
  s=reset(g,r,ood); F=[];H=[];PA=[];PR=[];Y=[]; pa=BOS; pr=0
  for t in range(pars(g,ood)['max']):
   F.append(raw(g,s,ood)); H.append(hand(g,s,ood)); PA.append(pa); PR.append(pr); y=oracle(g,s,ood); Y.append(ai(y)); act=y if r.random()>explore else int(r.choice(A)); s,pr,d,_=step(g,s,act,ood); pa=ai(act)
   if d: break
  F=np.array(F,dtype=np.float32);H=np.array(H,dtype=np.float32);PA=np.array(PA,dtype=np.int64);PR=np.array(PR,dtype=np.float32);Y=np.array(Y,dtype=np.int64)
  for st in range(0,len(Y),ctx):
   en=min(len(Y),st+ctx); L=en-st
   def pad(x,shape,val=0): z=np.full(shape,val,dtype=x.dtype);z[:L]=x[st:en];return z
   out.append((pad(F,(ctx,8)),pad(H,(ctx,48)),pad(PA,(ctx,),BOS),pad(PR,(ctx,)),pad(Y,(ctx,)),np.r_[np.ones(L),np.zeros(ctx-L)].astype(np.float32)))
   if en==len(Y): break
 return out

class OpAE(nn.Module):
 def __init__(self):
  super().__init__(); self.ld=24; self.ph=nn.Sequential(nn.Linear(8,64),nn.SiLU(),nn.Linear(64,8));self.ta=nn.Sequential(nn.Linear(8,64),nn.SiLU(),nn.Linear(64,8));self.dec=nn.Sequential(nn.Linear(24,64),nn.SiLU(),nn.Linear(64,8));self.ops=nn.ModuleList([nn.Linear(24,24) for _ in range(3)])
  for o in self.ops: nn.init.eye_(o.weight);nn.init.zeros_(o.bias)
 def enc(self,x):
  q=self.ph(x);return torch.cat([torch.sin(q),torch.cos(q),torch.tanh(self.ta(x))],-1)
 def trans(self,z,a):
  o=torch.empty_like(z)
  for i in range(3):
   m=a==i
   if m.any(): o[m]=self.ops[i](z[m])
  return o

def train_op(train_games,cfg,dev,sd):
 seed(sd);r=np.random.default_rng(sd);X=[];U=[];Z=[]
 for g in train_games:
  s=reset(g,r)
  for _ in range(cfg['op_pairs']):
   if r.random()<.08:s=reset(g,r)
   a=int(r.choice(A));ns,_,d,_=step(g,s,a);X.append(raw(g,s));U.append(ai(a));Z.append(raw(g,ns));s=reset(g,r) if d else ns
 X=torch.tensor(np.array(X),dtype=torch.float32);U=torch.tensor(U,dtype=torch.long);Z=torch.tensor(np.array(Z),dtype=torch.float32);dl=DataLoader(TensorDataset(X,U,Z),cfg['batch'],shuffle=True);m=OpAE().to(dev);opt=torch.optim.AdamW(m.parameters(),lr=1.5e-3)
 for _ in range(cfg['op_epochs']):
  for x,a,z2 in dl:
   x,a,z2=x.to(dev),a.to(dev),z2.to(dev);q=m.enc(x);q2=m.enc(z2);qp=m.trans(q,a);loss=((m.dec(q)-x)**2).mean()+1.6*((qp-q2)**2).mean()+.6*((m.dec(qp)-z2)**2).mean();opt.zero_grad();loss.backward();opt.step()
 return m.cpu()
def op_metric(m,g,n,sd,dev,ood=False):
 r=np.random.default_rng(sd);X=[];U=[];Z=[];s=reset(g,r,ood)
 for _ in range(n):
  a=int(r.choice(A));ns,_,d,_=step(g,s,a,ood);X.append(raw(g,s,ood));U.append(ai(a));Z.append(raw(g,ns,ood));s=reset(g,r,ood) if d else ns
 with torch.no_grad():
  x=torch.tensor(np.array(X)).to(dev);u=torch.tensor(U).to(dev);z2=torch.tensor(np.array(Z)).to(dev);mm=m.to(dev);q=mm.enc(x);qp=mm.trans(q,u);return float(((qp-mm.enc(z2))**2).mean().cpu()),float(((mm.dec(qp)-z2)**2).mean().cpu())

class RT(nn.Module):
 def __init__(self,d,ctx,dm=96,layers=2):
  super().__init__();self.ctx=ctx;self.sp=nn.Sequential(nn.Linear(d,dm),nn.LayerNorm(dm),nn.SiLU());self.ae=nn.Embedding(4,dm);self.rp=nn.Linear(1,dm);self.pe=nn.Parameter(torch.randn(1,ctx,dm)*.01);self.gru=nn.GRU(dm,dm,batch_first=True);el=nn.TransformerEncoderLayer(dm,4,dm*2,.05,batch_first=True,norm_first=True,activation='gelu');self.tr=nn.TransformerEncoder(el,layers);self.h=nn.Linear(dm,3)
 def forward(self,x,a,r):
  T=x.shape[1];z=self.sp(x)+self.ae(a)+self.rp(r[...,None])+self.pe[:,:T];z,_=self.gru(z);mask=torch.triu(torch.full((T,T),float('-inf'),device=x.device),1);return self.h(self.tr(z,mask=mask))
def dataset(items,rep,op=None):
 F=[];PA=[];PR=[];Y=[];M=[]
 with torch.no_grad():
  for x,h,a,r,y,m in items:
   if rep=='raw':f=x
   elif rep=='hand':f=h
   else:f=op.enc(torch.tensor(x,dtype=torch.float32)).numpy()
   F.append(f);PA.append(a);PR.append(r);Y.append(y);M.append(m)
 return TensorDataset(torch.tensor(np.array(F),dtype=torch.float32),torch.tensor(np.array(PA),dtype=torch.long),torch.tensor(np.array(PR),dtype=torch.float32),torch.tensor(np.array(Y),dtype=torch.long),torch.tensor(np.array(M),dtype=torch.float32))
def train_pol(items,val,rep,op,cfg,dev,sd):
 seed(sd);ds=dataset(items,rep,op);vd=dataset(val,rep,op);d={'raw':8,'hand':48,'auto':24}[rep];m=RT(d,cfg['ctx'],cfg['dm'],cfg['layers']).to(dev);opt=torch.optim.AdamW(m.parameters(),lr=8e-4,weight_decay=1e-4);dl=DataLoader(ds,cfg['batch'],shuffle=True);vl=DataLoader(vd,cfg['batch']);best=None;bv=1e9;pat=0
 for ep in range(cfg['epochs']):
  m.train()
  for x,a,r,y,ms in dl:
   x,a,r,y,ms=x.to(dev),a.to(dev),r.to(dev),y.to(dev),ms.to(dev);L=nn.functional.cross_entropy(m(x,a,r).reshape(-1,3),y.reshape(-1),reduction='none').reshape_as(ms);loss=(L*ms).sum()/ms.sum();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.5);opt.step()
  m.eval();ls=[]
  with torch.no_grad():
   for x,a,r,y,ms in vl:
    x,a,r,y,ms=x.to(dev),a.to(dev),r.to(dev),y.to(dev),ms.to(dev);L=nn.functional.cross_entropy(m(x,a,r).reshape(-1,3),y.reshape(-1),reduction='none').reshape_as(ms);ls.append(float(((L*ms).sum()/ms.sum()).cpu()))
  v=np.mean(ls)
  if v<bv-1e-4:bv=v;best=copy.deepcopy(m.state_dict());pat=0
  else:
   pat+=1
   if pat>=4:break
 m.load_state_dict(best);return m

def feat(rep,g,s,op,ood=False):
 if rep=='raw':return raw(g,s,ood)
 if rep=='hand':return hand(g,s,ood)
 with torch.no_grad():return op.enc(torch.tensor(raw(g,s,ood),dtype=torch.float32)[None]).squeeze().numpy()
def probs(c,g,s,hist,dev,ood=False):
 rep,sd,m,op=c; k=(rep,sd); hist.setdefault(k,[]).append(feat(rep,g,s,op,ood));x=np.array(hist[k][-m.ctx:]);a=np.array(hist['a'][-m.ctx:]);r=np.array(hist['r'][-m.ctx:]);
 with torch.no_grad():return torch.softmax(m(torch.tensor(x,dtype=torch.float32)[None].to(dev),torch.tensor(a,dtype=torch.long)[None].to(dev),torch.tensor(r,dtype=torch.float32)[None].to(dev))[0,-1],-1).cpu().numpy()
def eval_states(cands,w,g,n,sd,dev,ood=False):
 r=np.random.default_rng(sd);ok=0;ce=[];br=[];rg=[]
 for _ in range(n):
  s=reset(g,r,ood)
  for j in range(int(r.integers(0,max(1,pars(g,ood)['max']//3)))):
   s,_,d,_=step(g,s,int(r.choice(A)),ood)
   if d:s=reset(g,r,ood);break
  ps=[]
  for c in cands: ps.append(probs(c,g,s,{'a':[BOS],'r':[0.]},dev,ood))
  p=np.sum(np.array(ps)*w[:,None],0);y=ai(oracle(g,s,ood));pred=int(p.argmax());ok+=pred==y;ce.append(-math.log(max(1e-8,p[y])));one=np.eye(3)[y];br.append(((p-one)**2).mean());sp,_,_,_=step(g,s,av(pred),ood);so,_,_,_=step(g,s,oracle(g,s,ood),ood);rg.append(max(0,progress(g,so,ood)-progress(g,sp,ood)))
 return ok/n,np.mean(ce),np.mean(br),np.mean(rg)
def rollout(cands,w,g,n,sd,dev,ood=False):
 r=np.random.default_rng(sd);S=[];R=[]
 for _ in range(n):
  s=reset(g,r,ood);hist={'a':[BOS],'r':[0.]};ret=0
  for t in range(pars(g,ood)['max']):
   p=np.sum(np.array([probs(c,g,s,hist,dev,ood) for c in cands])*w[:,None],0);ix=int(p.argmax());s,re,d,su=step(g,s,av(ix),ood);ret+=re;hist['a'].append(ix);hist['r'].append(re)
   if d:S.append(float(su));R.append(ret);break
  else:S.append(0);R.append(ret)
 return np.mean(S),np.mean(R)
def hus(cands,train_games,cfg,dev,sd):
 r=np.random.default_rng(sd);anchors=[]
 for g in train_games:
  for _ in range(cfg['anchors']//3):anchors.append((g,reset(g,r)))
 D=[];O=[]
 for c in cands:
  vec=[];er=[];bi=[];re=[]
  for g,s in anchors:
   p=probs(c,g,s,{'a':[BOS],'r':[0.]},dev);vec+=list(p);y=ai(oracle(g,s));er.append(p.argmax()!=y);bi.append(((p-np.eye(3)[y])**2).mean());sp,_,_,_=step(g,s,av(int(p.argmax())));so,_,_,_=step(g,s,oracle(g,s));re.append(max(0,progress(g,so)-progress(g,sp)))
  D.append(vec);O.append([np.mean(er),np.mean(bi),np.mean(re)])
 O=np.array(O);lo=np.percentile(O,5,0);hi=np.percentile(O,95,0);ON=np.clip((O-lo)/np.where(hi-lo>1e-8,hi-lo,1),-1,2);D=np.array(D);D-=D.mean(1,keepdims=True);D/=np.linalg.norm(D,axis=1,keepdims=True)+1e-8;dz=np.linalg.norm(D[:,None]-D[None,:],axis=-1)/math.sqrt(D.shape[1]);do=np.linalg.norm(ON[:,None]-ON[None,:],axis=-1)/math.sqrt(3);sz=np.median(dz[np.triu_indices(len(cands),1)]);so=np.median(do[np.triu_indices(len(cands),1)]);K=np.exp(-.5*(dz/max(sz,1e-6))**2-1.5*.5*(do/max(so,1e-6))**2);np.fill_diagonal(K,0);deg=K.sum(1);K/=np.sqrt((deg[:,None]+1e-8)*(deg[None,:]+1e-8));E=ON@np.array([.55,.25,.2]);E=(E-E.min())/max(1e-8,E.max()-E.min());H=np.diag(E)-.55*K;la,V=np.linalg.eigh(H);q=V@(np.exp(-4*(la-la.min()))*(V.T@(np.ones(len(cands))/math.sqrt(len(cands)))));q=q*q;q/=q.sum();sc=np.exp(-4*E);sc/=sc.sum();top=np.zeros(len(cands));top[E.argmin()]=1;return q,sc,top,E,O,len(cands)*len(anchors)*3

def writecsv(p,rows):
 ks=sorted({k for x in rows for k in x});f=open(p,'w',newline='');w=csv.DictWriter(f,ks);w.writeheader();w.writerows(rows);f.close()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',default='smoke');ap.add_argument('--out',default='/tmp/hg');ap.add_argument('--repo',default='');z=ap.parse_args();full=z.mode=='full';cfg=dict(ctx=16 if full else 10,train=360 if full else 35,val=80 if full else 12,op_pairs=2200 if full else 120,op_epochs=20 if full else 2,batch=128 if full else 64,epochs=16 if full else 2,dm=96 if full else 48,layers=2 if full else 1,seeds=[7,17,27] if full else [7],states=1000 if full else 50,roll=80 if full else 6,anchors=252 if full else 30,op_eval=1200 if full else 80)
 out=Path(z.out);shutil.rmtree(out,ignore_errors=True);out.mkdir(parents=True);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');t0=time.time();rows=[];hrows=[];orows=[]
 for fi,hold in enumerate(GAMES):
  if not full and fi>0:break
  train_games=[g for g in GAMES if g!=hold];r=np.random.default_rng(1000+fi);tr=[];va=[]
  for g in train_games:tr+=collect(g,cfg['train'],r,cfg['ctx'],.18);va+=collect(g,cfg['val'],r,cfg['ctx'],.05)
  op=train_op(train_games,cfg,dev,500+fi);dm,pm=op_metric(op,hold,cfg['op_eval'],700+fi,dev);dmo,pmo=op_metric(op,hold,cfg['op_eval'],800+fi,dev,True);orows.append(dict(holdout=hold,latent_dyn_mse=dm,state_pred_mse=pm,ood_latent_dyn_mse=dmo,ood_state_pred_mse=pmo))
  C=[]
  for rep in ('raw','hand','auto'):
   for sd in cfg['seeds']:
    m=train_pol(tr,va,rep,op if rep=='auto' else None,cfg,dev,sd+100*fi).to(dev).eval();C.append((rep,sd,m,op if rep=='auto' else None));print('trained',hold,rep,sd,flush=True)
  q,sc,tp,E,O,calls=hus(C,train_games,cfg,dev,900+fi)
  for i,c in enumerate(C):hrows.append(dict(holdout=hold,rep=c[0],seed=c[1],qhro=q[i],scalar=sc[i],top=tp[i],energy=E[i],oracle_error=O[i,0],oracle_brier=O[i,1],oracle_regret=O[i,2],oracle_calls=calls,candidates=len(C)))
  methods=[]
  for i,c in enumerate(C):w=np.zeros(len(C));w[i]=1;methods.append((c[0],c[1],w))
  for sd in cfg['seeds']:methods += [('hus_qhro',sd,q),('scalar_ensemble',sd,sc),('scalar_top',sd,tp)]
  for name,sd,w in methods:
   for dist,ood in [('id',False),('ood',True)]:
    ac,ce,br,rg=eval_states(C,w,hold,cfg['states'],2000+fi+sd+(100 if ood else 0),dev,ood);su,re=rollout(C,w,hold,cfg['roll'],3000+fi+sd+(100 if ood else 0),dev,ood);rows.append(dict(holdout=hold,distribution=dist,method=name,seed=sd,action_acc=ac,nll=ce,brier=br,progress_regret=rg,rollout_success=su,return_mean=re))
  print('fold done',hold,flush=True)
 writecsv(out/'raw.csv',rows);writecsv(out/'hus.csv',hrows);writecsv(out/'operator.csv',orows)
 summ=[]
 for k in sorted(set((x['holdout'],x['distribution'],x['method']) for x in rows)):
  rr=[x for x in rows if (x['holdout'],x['distribution'],x['method'])==k];summ.append(dict(holdout=k[0],distribution=k[1],method=k[2],n=len(rr),action_acc=np.mean([x['action_acc'] for x in rr]),rollout_success=np.mean([x['rollout_success'] for x in rr]),return_mean=np.mean([x['return_mean'] for x in rr]),nll=np.mean([x['nll'] for x in rr])))
 writecsv(out/'summary.csv',summ);cfg['runtime']=time.time()-t0;cfg['device']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU';(out/'config.json').write_text(json.dumps(cfg,indent=2))
 md=['# HyperGame G1 HF','',f"Runtime: {cfg['runtime']/60:.1f} min; device: {cfg['device']}",'',f"Protocol: one recurrent Transformer architecture; train on 3 games, hold out the 4th. HUS uses {3*len(cfg['seeds'])} candidates and exactly 3 weak oracles on {cfg['anchors']} shared anchor states.",'','|holdout|method|action|rollout|return|','|---|---|---:|---:|---:|']
 for x in summ:
  if x['distribution']=='id':md.append(f"|{x['holdout']}|{x['method']}|{x['action_acc']:.3f}|{x['rollout_success']:.3f}|{x['return_mean']:.3f}|")
 md+=['','## Gates','- hand geometry excludes predicted impact/oracle actions;','- automatic geometry sees only unlabeled (s,a,s_next) triples from training games;','- scalar and QHRO/HUS receive identical candidates and oracle calls;','- publication-level positive result requires HUS > scalar_ensemble on held-out games.']
 (out/'REPORT.md').write_text('\n'.join(md));(out/'SOURCE.py').write_text(globals().get('SOURCE_TEXT','# source executed inline'))
 print('\n'.join(md),flush=True)
 import io,zipfile,base64
 buf=io.BytesIO()
 with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as zz:
  for fp in sorted(out.iterdir()): zz.write(fp,fp.name)
 print('ARTIFACT_B64='+base64.b64encode(buf.getvalue()).decode(),flush=True)
 if z.repo:
  from huggingface_hub import HfApi
  api=HfApi(token=os.environ['HF_TOKEN']);api.create_repo(z.repo,repo_type='dataset',exist_ok=True);api.upload_folder(repo_id=z.repo,repo_type='dataset',folder_path=str(out));print('HUB https://huggingface.co/datasets/'+z.repo,flush=True)
if __name__=='__main__':main()
