import sys, json, math, zipfile, tempfile, shutil, base64
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

URLS = sys.argv[1:4]
OUT = Path('/tmp/hdv2-analysis'); OUT.mkdir(exist_ok=True)
SEED = 20260716
rng = np.random.default_rng(SEED)

def download(url, path):
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(path,'wb') as f:
            for chunk in r.iter_content(1<<20):
                if chunk: f.write(chunk)

def upload(path):
    with open(path,'rb') as f:
        r=requests.post('https://tmpfiles.org/api/v1/upload', files={'file':(path.name,f,'application/zip')}, timeout=600)
    r.raise_for_status(); return r.json()['data']['url']

def fit_probe(X,y,C):
    m=Pipeline([('scale',StandardScaler()),('lr',LogisticRegression(C=C,max_iter=4000,class_weight='balanced',solver='liblinear'))])
    m.fit(X,y); return m

def make_folds(df,k=5):
    folds=[[] for _ in range(k)]
    rr=np.random.default_rng(SEED)
    for dom in sorted(df.domain.unique()):
        ids=np.array(sorted(df.loc[df.domain==dom,'task_id'].unique()),dtype=object)
        rr.shuffle(ids)
        for i,x in enumerate(ids): folds[i%k].append(x)
    return folds

def group_inner_splits(task_ids,k=3):
    ids=np.array(sorted(set(task_ids)),dtype=object)
    rr=np.random.default_rng(SEED+17); rr.shuffle(ids)
    parts=np.array_split(ids,k)
    for p in parts:
        va=set(p.tolist()); tr=np.array([x not in va for x in task_ids]); vv=np.array([x in va for x in task_ids])
        yield tr,vv

def safe_auc(y,p):
    return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float('nan')

def task_pick(g,score_col):
    return int(g.loc[g[score_col].idxmax(),'correct'])

def majority_pick(g):
    counts=g.choice.value_counts(); mx=counts.max(); candidates=set(counts[counts==mx].index.tolist())
    gg=g[g.choice.isin(candidates)].groupby('choice').margin.sum()
    ch=int(gg.idxmax()); return int(ch==int(g.answer.iloc[0]))

def bootstrap_task(values,B=4000,seed=1):
    a=np.asarray(values,float); rr=np.random.default_rng(seed)
    q=np.array([a[rr.integers(0,len(a),len(a))].mean() for _ in range(B)])
    return [float(a.mean()),float(np.quantile(q,.025)),float(np.quantile(q,.975))]

frames=[]; embs=[]; shard_meta=[]
for ui,url in enumerate(URLS):
    zp=OUT/f'shard{ui}.zip'; download(url,zp)
    d=OUT/f'shard{ui}'; d.mkdir(exist_ok=True)
    with zipfile.ZipFile(zp) as z: z.extractall(d)
    f=pd.read_parquet(d/'traces.parquet'); z=np.load(d/'emb_proj.npz'); x=z['x'].astype('float32'); layers=z['layers'].tolist()
    if len(f)!=len(x): raise RuntimeError('row/embedding mismatch')
    frames.append(f); embs.append(x)
    shard_meta.append(json.loads((d/'shard_summary.json').read_text()))
F=pd.concat(frames,ignore_index=True); X=np.concatenate(embs,axis=0); y=F.correct.to_numpy(int); groups=F.task_id.to_numpy()
np.savez_compressed(OUT/'merged_emb_proj.npz',x=X,layers=np.asarray(layers))
F.to_parquet(OUT/'merged_traces.parquet')
folds=make_folds(F,5)
oof=np.full(len(F),np.nan); oof_layer={l:np.full(len(F),np.nan) for l in layers}; fold_records=[]; path_records=[]
Cs=[.01,.1,1,10]

for fi,test_ids in enumerate(folds):
    te=F.task_id.isin(test_ids).to_numpy(); tr=~te
    train_idx=np.where(tr)[0]; test_idx=np.where(te)[0]
    best=(-1,None,None)
    for li,l in enumerate(layers):
        for C in Cs:
            vals=[]
            for itr,iva in group_inner_splits(F.loc[tr,'task_id'].to_numpy(),3):
                a=train_idx[itr]; b=train_idx[iva]
                if len(np.unique(y[a]))<2 or len(np.unique(y[b]))<2: continue
                m=fit_probe(X[a,li],y[a],C); vals.append(safe_auc(y[b],m.predict_proba(X[b,li])[:,1]))
            v=float(np.nanmean(vals)) if vals else -1
            if v>best[0]: best=(v,li,C)
    _,li,C=best
    m=fit_probe(X[tr,li],y[tr],C); oof[te]=m.predict_proba(X[te,li])[:,1]
    fold_records.append({'fold':fi,'layer':layers[li],'C':C,'inner_auc':best[0],'test_auc':safe_auc(y[te],oof[te])})
    for lj,l in enumerate(layers):
        mm=fit_probe(X[tr,lj],y[tr],.1); oof_layer[l][te]=mm.predict_proba(X[te,lj])[:,1]
    # Train-only OOF values to tune path penalties without leaking test tasks.
    train_oof=np.full(tr.sum(),np.nan); local_ids=F.loc[tr,'task_id'].to_numpy(); local_y=y[tr]; local_X=X[tr,li]
    for ia,ib in group_inner_splits(local_ids,3):
        mm=fit_probe(local_X[ia],local_y[ia],C); train_oof[ib]=mm.predict_proba(local_X[ib])[:,1]
    TT=F.loc[tr].copy(); TT['probe_tmp']=train_oof
    mu=TT[['tokens','curvature','entropy']].mean(); sd=TT[['tokens','curvature','entropy']].std().replace(0,1)
    for c in ['tokens','curvature','entropy']: TT['z_'+c]=(TT[c]-mu[c])/sd[c]
    grids=[0,.03,.07,.15,.3]; bestp=(-1,(0,0,0))
    for a in grids:
      for b in grids:
       for c in grids:
        TT['ps']=TT.probe_tmp-a*TT.z_tokens-b*TT.z_curvature-c*TT.z_entropy
        acc=np.mean([task_pick(g,'ps') for _,g in TT.groupby('task_id')])
        if acc>bestp[0]: bestp=(acc,(a,b,c))
    a,b,c=bestp[1]
    VV=F.loc[te].copy(); VV['probe_oof']=oof[te]
    for col in ['tokens','curvature','entropy']: VV['z_'+col]=(VV[col]-mu[col])/sd[col]
    VV['path_oof']=VV.probe_oof-a*VV.z_tokens-b*VV.z_curvature-c*VV.z_entropy
    for idx,row in VV.iterrows(): path_records.append({'row':int(idx),'path_oof':float(row.path_oof),'fold':fi,'a':a,'b':b,'c':c})

F['probe_oof']=oof
PTH=pd.DataFrame(path_records).set_index('row'); F['path_oof']=PTH.loc[F.index,'path_oof'].to_numpy()
for l in layers: F[f'probe_layer_{l}']=oof_layer[l]
F.to_parquet(OUT/'oof_predictions.parquet')
pd.DataFrame(fold_records).to_csv(OUT/'folds.csv',index=False)
PTH.reset_index().to_csv(OUT/'path_params_and_scores.csv',index=False)

# task-level evaluation
rows=[]
for tid,g in F.groupby('task_id'):
    ans=int(g.answer.iloc[0]); first=g.sort_values('rollout').iloc[0]
    rows.append({'task_id':tid,'domain':g.domain.iloc[0],
      'raw':int(first.correct),'margin':task_pick(g,'margin'),'confidence':task_pick(g,'confidence'),
      'majority':majority_pick(g),'probe':task_pick(g,'probe_oof'),'path':task_pick(g,'path_oof'),
      'oracle':int(g.correct.max())})
T=pd.DataFrame(rows); T.to_csv(OUT/'task_methods.csv',index=False)
method_rows=[]
for i,meth in enumerate(['raw','margin','confidence','majority','probe','path','oracle']):
    est,lo,hi=bootstrap_task(T[meth],4000,SEED+i)
    method_rows.append({'method':meth,'estimate':est,'low95':lo,'high95':hi})
M=pd.DataFrame(method_rows); M.to_csv(OUT/'methods.csv',index=False)

layer_rows=[]
for l in layers: layer_rows.append({'layer':l,'oof_auc':safe_auc(y,F[f'probe_layer_{l}'])})
L=pd.DataFrame(layer_rows); L.to_csv(OUT/'layer_auc.csv',index=False)

# low-rank centered success/failure structure
Z=X.reshape(len(X),-1); C=np.vstack([Z[y==1]-Z[y==1].mean(0),Z[y==0]-Z[y==0].mean(0)])
s=np.linalg.svd(C,compute_uv=False); ev=np.cumsum(s*s)/np.sum(s*s); rank90=int(np.searchsorted(ev,.9)+1)
pd.DataFrame({'singular_value':s,'cum_variance':ev}).to_csv(OUT/'singular_values.csv',index=False)

# curvature paired task statistic on tasks with both outcomes
cd=[]
for tid,g in F.groupby('task_id'):
    if g.correct.nunique()==2: cd.append(float(g.loc[g.correct==0,'curvature'].mean()-g.loc[g.correct==1,'curvature'].mean()))
curv_ci=bootstrap_task(cd,4000,SEED+99) if cd else [float('nan')]*3

rollout_by_domain=F.groupby('domain').correct.mean().to_dict()
task_by_domain={d:{m:float(g[m].mean()) for m in ['raw','margin','confidence','majority','probe','path','oracle']} for d,g in T.groupby('domain')}
summary={'shards':shard_meta,'n_tasks':len(T),'n_trajectories':len(F),'chance':1/6,
 'rollout_accuracy':float(F.correct.mean()),'rollout_accuracy_by_domain':rollout_by_domain,
 'methods':M.to_dict('records'),'task_methods_by_domain':task_by_domain,
 'probe_oof_auc':safe_auc(y,F.probe_oof),'layer_auc':L.to_dict('records'),
 'rank90_projected':rank90,'curvature_failure_minus_success_task_paired':{'n_tasks':len(cd),'estimate':curv_ci[0],'low95':curv_ci[1],'high95':curv_ci[2]},
 'folds':fold_records}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))

best_nonoracle=M[M.method!='oracle'].sort_values('estimate').iloc[-1]
report=f'''# HyperDiscovery v2 — Combined analysis\n\n- Tasks: **{len(T)}**\n- Trajectories: **{len(F)}**\n- Rollout accuracy: **{F.correct.mean():.3%}** (chance 16.667%)\n- OOF probe AUROC: **{summary['probe_oof_auc']:.4f}**\n- Projected rank for 90% variance: **{rank90}**\n- Best non-oracle task method: **{best_nonoracle.method} = {best_nonoracle.estimate:.3%}** (95% task bootstrap {best_nonoracle.low95:.3%}–{best_nonoracle.high95:.3%})\n\n## Methods\n\n{M.to_markdown(index=False)}\n\n## Rollout accuracy by domain\n\n{pd.DataFrame([rollout_by_domain]).T.rename(columns={{0:'accuracy'}}).to_markdown()}\n\n## Layer probes\n\n{L.to_markdown(index=False)}\n\n## Interpretation\n\nChoice-logit scoring removes the parser confound. The oracle row is only an upper bound: it assumes a perfect verifier can identify a successful rollout. Probe and path rows are strictly out-of-fold at task level.\n'''
(OUT/'REPORT.md').write_text(report)
html='<html><head><meta charset="utf-8"><style>body{font-family:Arial;max-width:1050px;margin:40px auto;background:#0d1324;color:#eef3ff;line-height:1.5}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #334;padding:9px}pre{white-space:pre-wrap;background:#171f35;padding:18px;border-radius:12px}</style></head><body><h1>HyperDiscovery v2 — Combined Analysis</h1><pre>'+json.dumps(summary,indent=2)+'</pre></body></html>'
(OUT/'REPORT.html').write_text(html)
full=Path('/tmp/hdv2-combined-analysis-full.zip'); compact=Path('/tmp/hdv2-combined-analysis-compact.zip')
with zipfile.ZipFile(full,'w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.iterdir(): z.write(p,p.name)
with zipfile.ZipFile(compact,'w',zipfile.ZIP_DEFLATED) as z:
    for n in ['summary.json','REPORT.md','REPORT.html','methods.csv','task_methods.csv','layer_auc.csv','folds.csv','singular_values.csv','path_params_and_scores.csv']:
        z.write(OUT/n,n)
print('ANALYSIS_SUMMARY='+json.dumps(summary),flush=True)
print('FULL_URL='+upload(full),flush=True)
print('COMPACT_URL='+upload(compact),flush=True)
print('COMPACT_B64_BEGIN')
b=base64.b64encode(compact.read_bytes()).decode()
for i in range(0,len(b),8000): print(b[i:i+8000])
print('COMPACT_B64_END')
