# /// script
# requires-python = ">=3.11"
# dependencies = ["torch","transformers>=5,<6","huggingface_hub>=1.0","accelerate>=1.2","safetensors>=0.4","numpy>=1.26","pandas>=2.2"]
# ///
from __future__ import annotations
import contextlib, hashlib, json, math, random
from dataclasses import dataclass
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID='harims95/LoopLM-135M-naive'; REV='a6288c7f257333634ef66dafecf966ad581d9316'; OUT=Path('/tmp/p17_path_transport_v3')
EPS=(.005,.008,.0125,.02,.032); LABELS=('A','B','C','D'); SEED=20260723; NQ=5; NEX=4
@dataclass(frozen=True)
class Mode:
    name:str; input_scale:float=1.; update_scale:float=1.; state_decay:float=1.
M={'il':Mode('il',.65,.78,.97),'ih':Mode('ih',1.25,.78,.97),'ul':Mode('ul',1.,.55,.97),'uh':Mode('uh',1.,1.,.97)}
PAIRS={'input':(M['il'],M['ih']),'update':(M['ul'],M['uh']),'input_x_update':(M['ih'],M['uh'])}
@contextlib.contextmanager
def fixed_rng(seed,device):
    ds=[device.index or torch.cuda.current_device()] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=ds):
        torch.manual_seed(seed)
        if device.type=='cuda': torch.cuda.manual_seed_all(seed)
        yield
def sseed(s): return int(hashlib.sha256(f'{SEED}|{s}'.encode()).hexdigest()[:8],16)%(2**31-1)
def prompts(n):
    out=[]; profs=('early','uniform','late')
    for vol in (0,1,2):
      for prof in profs:
        r=random.Random(SEED+101*vol+17*profs.index(prof)); st=r.randrange(4); ch=[] if vol==0 else ([13] if vol==1 else [10,13]); lines=[]
        for t in range(1,17):
          if t in ch: st=r.choice([q for q in range(4) if q!=st])
          x=(t-1)/15; p=.75-.30*x if prof=='early' else (.45+.30*x if prof=='late' else .60)
          if r.random()<p:
            rep=st if r.random()<.82 else r.choice([q for q in range(4) if q!=st]); lines.append(f't={t:02d}: noisy sensor reports state {LABELS[rep]}.')
          else: lines.append(f't={t:02d}: no informative observation.')
        text='You monitor a hidden state chosen from A, B, C, D. Sensor reports are independently correct with probability 82%. The hidden state may change over time. Infer the state active at the FINAL time.\n\n'+'\n'.join(lines)+'\n\nReturn exactly one label: A, B, C, or D.\nAnswer:'
        out.append({'id':f'p17-v{vol}-{prof}','vol':vol,'prof':prof,'prompt':text})
    return out[:n]
def inner(model):
    x=model.model
    for a in ('_run_prelude','loop','_run_coda','_h0','lm_head'):
        if not hasattr(x,a): raise RuntimeError('LoopLM internals changed')
    return x
def hzero(x,b,l,d,dt,seed):
    with fixed_rng(seed,d): return x._h0(b,l,d,dt)
def raw_step(x,h,h0,e,cos,sin,m):
    cand=x.loop(h,e*float(m.input_scale),cos,sin); upd=h+float(m.update_scale)*(cand-h)
    return float(m.state_decay)*upd+(1-float(m.state_decay))*h0
def schedule(x,h0,e,cos,sin,modes,eps):
    h=h0.clone()
    for m in modes:
        raw=raw_step(x,h,h0,e,cos,sin,m); h=h+eps*(raw-h)
    return h
def label_ids(tok):
    out=[]
    for lab in LABELS:
      for s in (' '+lab,lab):
        ids=tok.encode(s,add_special_tokens=False)
        if len(ids)==1: out.append(ids[0]); break
      else: raise RuntimeError(lab)
    return out
def readout_fn(x,cos,sin,lids):
    idx=torch.tensor(lids,device=cos.device)
    def f(h):
        z=x.lm_head(x._run_coda(h,cos,sin))[:,-1,:]
        return z.index_select(-1,idx).squeeze(0).float()
    return f
def jac_rows(f,h):
    h=h.detach().requires_grad_(True); z=f(h); rows=[]
    for i in range(z.numel()):
        g=torch.autograd.grad(z[i],h,retain_graph=i+1<z.numel(),create_graph=False)[0]
        rows.append(g.detach().flatten().float())
    return torch.stack(rows)
def jv(j,v): return j@v.detach().flatten().float()
def basis(j,tol=1e-7):
    _,s,vh=torch.linalg.svd(j,full_matrices=False); rank=int((s>tol*s.max()).sum()) if s.numel() else 0
    return vh[:rank].T.contiguous()
def subspace(j0,j1):
    q0,q1=basis(j0),basis(j1)
    if q0.shape[1]==0 or q1.shape[1]==0: return {'angle':float('nan'),'proj':float('nan')}
    sv=torch.linalg.svdvals(q0.T@q1).clamp(0,1); ang=torch.acos(sv)*180/math.pi
    r0,r1=q0.shape[1],q1.shape[1]
    proj_sq=max(float(r0+r1-2*torch.sum(sv*sv).item()),0.0)
    return {'angle':float(ang.mean()),'proj':math.sqrt(proj_sq)}
def fit(df,m):
    x=np.log(df.epsilon.to_numpy(float)); y=np.log(np.maximum(df[m].to_numpy(float),1e-30)); c=np.polyfit(x,y,1); pred=np.polyval(c,x); den=max(float(((y-y.mean())**2).sum()),1e-30)
    return {'slope':float(c[0]),'r2':1-float(((y-pred)**2).sum())/den}
def main():
    OUT.mkdir(parents=True,exist_ok=True); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.type=='cuda':
        torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends.cuda,'enable_cudnn_sdp'): torch.backends.cuda.enable_cudnn_sdp(False)
    tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=REV,trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=REV,trust_remote_code=True,dtype=torch.float32,low_cpu_mem_usage=True).to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)
    x=inner(model); lids=label_ids(tok); nodes,w=np.polynomial.legendre.leggauss(NQ); nodes=(nodes+1)/2; w=w/2; rows=[]
    for ex in prompts(NEX):
      ids=tok(ex['prompt'],return_tensors='pt',truncation=True,max_length=320)['input_ids'].to(dev)
      with torch.no_grad(): e,cos,sin=x._run_prelude(ids); h0=hzero(x,1,ids.shape[1],dev,e.dtype,sseed(ex['id']))
      f=readout_fn(x,cos,sin,lids)
      for pn,(a,b) in PAIRS.items():
       for eps in EPS:
        with torch.no_grad(): hab=schedule(x,h0,e,cos,sin,(a,b),eps); hba=schedule(x,h0,e,cos,sin,(b,a),eps); exact=(f(hab)-f(hba)).detach().float()
        delta=(hab-hba).detach(); mid=(hab+hba).detach()/2; positions=[hba,mid,hab]+[(hba+float(s)*delta).detach() for s in nodes]
        jacs=[jac_rows(f,p) for p in positions]; d0,dm,d1=[jv(j,delta) for j in jacs[:3]]; integ=torch.zeros_like(exact)
        for wi,ji in zip(w,jacs[3:]): integ+=float(wi)*jv(ji,delta)
        corr=integ-dm; en=float(torch.linalg.vector_norm(exact)); dn=float(torch.linalg.vector_norm(delta)); sm=subspace(jacs[0],jacs[2])
        rows.append({'id':ex['id'],'vol':ex['vol'],'prof':ex['prof'],'pair':pn,'epsilon':eps,'pre_delta':dn,'exact':en,'integrated':float(torch.linalg.vector_norm(integ)),'static_mid':float(torch.linalg.vector_norm(dm)),'transport_correction':float(torch.linalg.vector_norm(corr)),'jacobian_drift_action':float(torch.linalg.vector_norm(d1-d0)),'integral_rel_error':float(torch.linalg.vector_norm(integ-exact))/max(en,1e-20),'static_rel_error':float(torch.linalg.vector_norm(dm-exact))/max(en,1e-20),'transport_fraction':float(torch.linalg.vector_norm(corr))/max(en,1e-20),'cos_integrated':float(torch.nn.functional.cosine_similarity(integ[None],exact[None])),'cos_static':float(torch.nn.functional.cosine_similarity(dm[None],exact[None])),'angle_deg':sm['angle'],'projector_distance':sm['proj']})
        del jacs
        if dev.type=='cuda': torch.cuda.empty_cache()
    raw=pd.DataFrame(rows); raw.to_csv(OUT/'raw.csv',index=False); means=raw.groupby(['pair','epsilon'],as_index=False).mean(numeric_only=True); means.to_csv(OUT/'means.csv',index=False)
    metrics=['pre_delta','exact','integrated','static_mid','transport_correction','jacobian_drift_action']; fits=[]
    for pn,g in means.groupby('pair'):
      for m in metrics: fits.append({'pair':pn,'metric':m,**fit(g,m)})
    agg=[]
    for pn,g in raw.groupby('pair'):
      agg.append({'pair':pn,'integral_rel_error':float(g.integral_rel_error.mean()),'static_rel_error':float(g.static_rel_error.mean()),'transport_fraction':float(g.transport_fraction.mean()),'cos_integrated':float(g.cos_integrated.mean()),'cos_static':float(g.cos_static.mean()),'angle_deg':float(g.angle_deg.mean()),'projector_distance':float(g.projector_distance.mean())})
    summary={'experiment':'P17 path-integrated Jacobian transport v3','status':'completed','model':MODEL_ID,'revision':REV,'device':str(dev),'dtype':str(next(model.parameters()).dtype),'n_examples':NEX,'quadrature_points':NQ,'epsilons':EPS,'fits':fits,'aggregate':agg,'note':'Projector distance computed from principal singular values; no HxH projector materialization.'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print('FINAL_P17_V3_SUMMARY_BEGIN'); print(json.dumps(summary,indent=2)); print('FINAL_P17_V3_SUMMARY_END')
if __name__=='__main__': main()
