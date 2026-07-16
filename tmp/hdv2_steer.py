import sys,json,math,random,re,zipfile,base64,time
from pathlib import Path
from collections import defaultdict
import numpy as np,pandas as pd,torch,requests
from transformers import AutoTokenizer,AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

PAGES=sys.argv[1:4]
MODEL='Qwen/Qwen3-4B-Instruct-2507'; SEED=20260716; LAYERS=[7,14,21,28,35]
OUT=Path('/tmp/hdv2-steering');OUT.mkdir(exist_ok=True)
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)

def resolve(page):
    t=requests.get(page,timeout=60).text
    return re.findall(r'href=[\"\'](https://tmpfiles.org/dl/[^\"\']+)',t)[0]
def download(page,path):
    u=resolve(page)
    with requests.get(u,stream=True,timeout=600) as r:
        r.raise_for_status()
        with open(path,'wb') as f:
            for c in r.iter_content(1<<20):
                if c:f.write(c)
def upload(p):
    with open(p,'rb') as f:r=requests.post('https://tmpfiles.org/api/v1/upload',files={'file':(p.name,f,'application/zip')},timeout=600)
    r.raise_for_status();return r.json()['data']['url']
def make_folds(F,k=5):
    folds=[[] for _ in range(k)];rr=np.random.default_rng(SEED)
    for dom in sorted(F.domain.unique()):
        ids=np.array(sorted(F.loc[F.domain==dom,'task_id'].unique()),dtype=object);rr.shuffle(ids)
        for i,x in enumerate(ids):folds[i%k].append(x)
    return folds
def fit_probe(X,y):
    m=Pipeline([('s',StandardScaler()),('m',LogisticRegression(C=.1,max_iter=3000,class_weight='balanced',solver='liblinear'))]);m.fit(X,y);return m
def chat(tok,t):
    return tok.apply_chat_template([{'role':'system','content':'Solve exact multiple-choice tasks. Reason carefully; a separate scorer chooses 0..5.'},{'role':'user','content':t['prompt']}],tokenize=False,add_generation_prompt=True)
def cp(a):
    m=min(map(len,a))
    for i in range(m):
        if len({x[i] for x in a})>1:return i
    return m
def score(model,tok,prefix,layer=None,vec=None,alpha=0.):
    seq=[tok(prefix+f' {i}',add_special_tokens=False).input_ids for i in range(6)];c=cp(seq);M=max(map(len,seq));pad=tok.pad_token_id
    x=torch.full((6,M),pad,dtype=torch.long,device=model.device);am=torch.zeros_like(x)
    for i,s in enumerate(seq):x[i,:len(s)]=torch.tensor(s,device=model.device);am[i,:len(s)]=1
    hook=None
    if layer is not None and alpha and vec is not None:
        def hk(mod,inp,out):
            if isinstance(out,tuple):
                z=out[0].clone();z[:,max(0,c-1),:]+=alpha*vec.to(z.dtype);return (z,)+out[1:]
            z=out.clone();z[:,max(0,c-1),:]+=alpha*vec.to(z.dtype);return z
        hook=model.model.layers[layer].register_forward_hook(hk)
    try:
        with torch.inference_mode():lp=torch.log_softmax(model(input_ids=x,attention_mask=am,use_cache=False).logits.float(),-1)
        out=np.array([sum(float(lp[i,j-1,s[j]]) for j in range(c,len(s))) for i,s in enumerate(seq)])
        return int(out.argmax()),out
    finally:
        if hook:hook.remove()
def boot_delta(df,a,b,B=4000):
    ids=df.task_id.unique();rr=np.random.default_rng(SEED+77);vals=[]
    d=dict(zip(df.task_id,df[a]-df[b]))
    arr=np.array([d[x] for x in ids],float)
    for _ in range(B):vals.append(arr[rr.integers(0,len(arr),len(arr))].mean())
    return [float(arr.mean()),float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]

frames=[];full=[];proj=[];taskmap={}
for i,p in enumerate(PAGES):
    z=OUT/f's{i}.zip';download(p,z);d=OUT/f's{i}';d.mkdir(exist_ok=True)
    with zipfile.ZipFile(z) as q:q.extractall(d)
    F=pd.read_parquet(d/'traces.parquet');frames.append(F)
    full.append(np.load(d/'emb_full.npz')['x'].astype('float32'));proj.append(np.load(d/'emb_proj.npz')['x'].astype('float32'))
    for line in (d/'tasks.jsonl').read_text().splitlines():
        t=json.loads(line);taskmap[t['id']]=t
F=pd.concat(frames,ignore_index=True);X=np.concatenate(full);Z=np.concatenate(proj);y=F.correct.to_numpy(int)
folds=make_folds(F);tok=AutoTokenizer.from_pretrained(MODEL);tok.pad_token=tok.pad_token or tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,device_map='auto');model.eval();H=model.config.hidden_size
results=[];foldmeta=[];start=time.time()
for fi,testids in enumerate(folds):
    print('STEER_FOLD',fi+1,'/5',flush=True)
    te=F.task_id.isin(testids).to_numpy();tr=~te
    # choose layer using train-only three-way task split on projected representations
    train_tasks=np.array(sorted(F.loc[tr,'task_id'].unique()),dtype=object);rr=np.random.default_rng(SEED+fi);rr.shuffle(train_tasks);parts=np.array_split(train_tasks,3)
    best=(-1,0)
    for li,l in enumerate(LAYERS):
        vals=[]
        for va in parts:
            vv=F.task_id.isin(va).to_numpy()&tr;tt=tr&~vv
            if len(np.unique(y[tt]))<2 or len(np.unique(y[vv]))<2:continue
            m=fit_probe(Z[tt,li],y[tt]);vals.append(roc_auc_score(y[vv],m.predict_proba(Z[vv,li])[:,1]))
        v=float(np.mean(vals))
        if v>best[0]:best=(v,li)
    li=best[1];layer=LAYERS[li]
    S=X[tr,li];yy=y[tr];direction=S[yy==1].mean(0)-S[yy==0].mean(0);direction/=max(np.linalg.norm(direction),1e-9)
    scale=float(np.std(S@direction));vec=torch.tensor(direction*scale,device=model.device)
    # orthogonal random control
    rd=np.random.default_rng(SEED+fi).normal(size=H).astype('float32');rd-=direction*(rd@direction);rd/=max(np.linalg.norm(rd),1e-9);rvec=torch.tensor(rd*scale,device=model.device)
    # train-only calibration, 6 task IDs per domain
    cal=[]
    for dom in sorted(F.domain.unique()):cal+=sorted(F.loc[tr&(F.domain.to_numpy()==dom),'task_id'].unique())[:6]
    alphas=[-2,-1,-.5,0,.5,1,2];acal=[]
    for alpha in alphas:
        ok=[]
        for tid in cal:
            r=F[(F.task_id==tid)&(F.rollout==0)].iloc[0];t=taskmap[tid];prefix=chat(tok,t)+r.completion.rstrip()+'\nFINAL:'
            ch,_=score(model,tok,prefix,layer,vec,alpha);ok.append(ch==t['answer'])
        acal.append((float(np.mean(ok)),alpha))
    alpha=sorted(acal,key=lambda x:(x[0],-abs(x[1])),reverse=True)[0][1]
    foldmeta.append({'fold':fi,'layer':layer,'inner_auc':best[0],'alpha':alpha,'calibration':acal})
    for tid in testids:
        r=F[(F.task_id==tid)&(F.rollout==0)].iloc[0];t=taskmap[tid];prefix=chat(tok,t)+r.completion.rstrip()+'\nFINAL:'
        raw,_=score(model,tok,prefix);st,_=score(model,tok,prefix,layer,vec,alpha);rc,_=score(model,tok,prefix,layer,rvec,alpha)
        results.append({'task_id':tid,'domain':t['domain'],'fold':fi,'layer':layer,'alpha':alpha,'answer':t['answer'],'raw_choice':raw,'steered_choice':st,'random_choice':rc,'raw':int(raw==t['answer']),'steered':int(st==t['answer']),'random':int(rc==t['answer'])})
    if torch.cuda.is_available():torch.cuda.empty_cache()
R=pd.DataFrame(results);R.to_csv(OUT/'steering_results.csv',index=False)
summary={'tasks':len(R),'raw_accuracy':float(R.raw.mean()),'steered_accuracy':float(R.steered.mean()),'random_accuracy':float(R.random.mean()),'steered_minus_raw':boot_delta(R,'steered','raw'),'random_minus_raw':boot_delta(R,'random','raw'),'by_domain':{d:{c:float(g[c].mean()) for c in ['raw','steered','random']} for d,g in R.groupby('domain')},'folds':foldmeta,'runtime':time.time()-start,'gpu':torch.cuda.get_device_name(0)}
(OUT/'steering_summary.json').write_text(json.dumps(summary,indent=2));(OUT/'REPORT.md').write_text('# HyperDiscovery v2 causal steering\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n')
archive=Path('/tmp/hdv2-steering-results.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.iterdir():
        if p.is_file():z.write(p,p.name)
print('STEERING_SUMMARY='+json.dumps(summary),flush=True);print('STEERING_URL='+upload(archive),flush=True)
print('STEERING_B64_BEGIN');b=base64.b64encode(archive.read_bytes()).decode();[print(b[i:i+8000]) for i in range(0,len(b),8000)];print('STEERING_B64_END')
