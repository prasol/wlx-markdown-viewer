import sys, json, time, math, random, base64, zipfile
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import torch, requests
from transformers import AutoTokenizer, AutoModelForCausalLM

DOMAIN = sys.argv[1]
assert DOMAIN in {"subset","det","tri"}
RUN = f"hdv2-shard-{DOMAIN}"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SEED = 20260714 + {"subset":1,"det":2,"tri":3}[DOMAIN]
N, R = 60, 8
LAYERS = [7,14,21,28,35]
PROJ = 96
OUT = Path("/tmp") / RUN
OUT.mkdir(exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def ntri(edges):
    s = {tuple(sorted(x)) for x in edges}
    return sum((a,b) in s and (a,c) in s and (b,c) in s
               for a,b,c in combinations(range(6),3))

def make_tasks():
    out=[]
    for i in range(N):
        if DOMAIN=="subset":
            while True:
                nums=[random.randint(-12,18) for _ in range(6)]
                ss=[]
                for r in range(1,7):
                    for ix in combinations(range(6),r):
                        ss.append(([nums[j] for j in ix], sum(nums[j] for j in ix)))
                random.shuffle(ss)
                true_vals,target=ss[0]
                ans=random.randrange(6)
                choices=[None]*6; choices[ans]=true_vals; used={target}
                for k in range(6):
                    if k==ans: continue
                    for vals,total in ss:
                        if total!=target and total not in used:
                            choices[k]=vals; used.add(total); break
                if all(x is not None for x in choices): break
            prompt=(f"Numbers: {nums}\nTarget: {target}\n"
                    + "\n".join(f"{k}: {choices[k]}" for k in range(6))
                    + "\nAnalyze all options carefully.")
        elif DOMAIN=="det":
            mats=[]; dets=[]
            while len(mats)<6:
                m=[[random.randint(-6,6),random.randint(-6,6)],
                   [random.randint(-6,6),random.randint(-6,6)]]
                d=m[0][0]*m[1][1]-m[0][1]*m[1][0]
                if d not in dets: mats.append(m); dets.append(d)
            ans=random.randrange(6)
            prompt=(f"Target determinant: {dets[ans]}\n"
                    + "\n".join(f"{k}: {mats[k]}" for k in range(6))
                    + "\nAnalyze all options carefully.")
        else:
            all_edges=list(combinations(range(6),2))
            while True:
                graphs=[]; counts=[]
                for _ in range(3000):
                    edges=[x for x in all_edges if random.random()<random.uniform(.15,.8)]
                    c=ntri(edges)
                    if c not in counts: graphs.append(edges); counts.append(c)
                    if len(graphs)==6: break
                if len(graphs)==6: break
            ans=random.randrange(6)
            prompt=(f"Target triangles: {counts[ans]}\n"
                    + "\n".join(f"{k}: {graphs[k]}" for k in range(6))
                    + "\nAnalyze all options carefully.")
        out.append({"id":f"{DOMAIN}-{i}","domain":DOMAIN,"prompt":prompt,"answer":ans})
    return out

def chat(tok,t):
    return tok.apply_chat_template([
        {"role":"system","content":"Solve exact multiple-choice tasks. Reason carefully; a separate scorer chooses 0..5."},
        {"role":"user","content":t["prompt"]}], tokenize=False, add_generation_prompt=True)

def common_prefix(seqs):
    m=min(map(len,seqs))
    for i in range(m):
        if len({x[i] for x in seqs})>1: return i
    return m

def batch_choice_scores(model,tok,prefixes):
    seqs=[]; owners=[]
    for r,prefix in enumerate(prefixes):
        opts=[tok(prefix+f" {i}",add_special_tokens=False).input_ids for i in range(6)]
        c=common_prefix(opts)
        for s in opts:
            seqs.append(s); owners.append((r,c))
    M=max(map(len,seqs)); pad=tok.pad_token_id
    x=torch.full((len(seqs),M),pad,dtype=torch.long,device=model.device)
    am=torch.zeros_like(x)
    for i,s in enumerate(seqs):
        x[i,:len(s)]=torch.tensor(s,device=model.device); am[i,:len(s)]=1
    with torch.inference_mode():
        lp=torch.log_softmax(model(input_ids=x,attention_mask=am,use_cache=False).logits.float(),-1)
    out=np.zeros((len(prefixes),6),dtype=np.float64)
    for i,s in enumerate(seqs):
        r,c=owners[i]; opt=i%6
        out[r,opt]=sum(float(lp[i,j-1,s[j]]) for j in range(c,len(s)))
    return out

def curvature(z):
    if len(z)<3: return 0.0
    v=np.diff(z,axis=0); u=v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-9)
    return float(np.linalg.norm(np.diff(u,axis=0),axis=1).mean())

def upload(path):
    with open(path,"rb") as f:
        r=requests.post("https://tmpfiles.org/api/v1/upload", files={"file":(path.name,f,"application/zip")}, timeout=600)
    r.raise_for_status(); return r.json()["data"]["url"]

tasks=make_tasks(); (OUT/"tasks.jsonl").write_text("\n".join(json.dumps(x) for x in tasks))
tok=AutoTokenizer.from_pretrained(MODEL); tok.pad_token=tok.pad_token or tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,device_map="auto"); model.eval()
H=model.config.hidden_size
P=np.random.default_rng(SEED).choice([-1,1],(len(LAYERS),H,PROJ)).astype("float32")/math.sqrt(PROJ)
rows=[]; full_emb=[]; proj_emb=[]; start=time.time()

for ti,t in enumerate(tasks):
    if ti%5==0: print("GEN",DOMAIN,ti+1,"/",N,flush=True)
    base=chat(tok,t); enc=tok(base,return_tensors="pt",add_special_tokens=False).to(model.device); pl=enc.input_ids.shape[1]
    with torch.inference_mode():
        seqs=model.generate(**enc,do_sample=True,temperature=.8,top_p=.92,max_new_tokens=112,num_return_sequences=R,
            pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    texts=[tok.decode(s[pl:],skip_special_tokens=True) for s in seqs]
    scores=batch_choice_scores(model,tok,[base+x.rstrip()+"\nFINAL:" for x in texts])
    masks=torch.ones_like(seqs)
    for ri,s in enumerate(seqs):
        tail=s[pl:]; eos=(tail==tok.eos_token_id).nonzero(as_tuple=False)
        if len(eos): masks[ri,pl+int(eos[0])+1:]=0
    with torch.inference_mode():
        hs=model(input_ids=seqs,attention_mask=masks,output_hidden_states=True,use_cache=False).hidden_states[1:]
    for ri in range(R):
        sc=scores[ri]; prob=np.exp(sc-sc.max()); prob/=prob.sum(); choice=int(sc.argmax()); order=np.argsort(sc)[::-1]
        valid=int(masks[ri,pl:].sum().item()); ff=[]; pp=[]; cc=[]
        for q,l in enumerate(LAYERS):
            z=hs[l][ri,pl:pl+valid].float().cpu().numpy()
            if not len(z): z=hs[l][ri,pl-1:pl].float().cpu().numpy()
            m=z.mean(0); ff.append(m.astype("float16")); pp.append((m@P[q]).astype("float32")); cc.append(curvature(z[::4]))
        full_emb.append(np.stack(ff)); proj_emb.append(np.stack(pp))
        rows.append({"task_id":t["id"],"domain":DOMAIN,"rollout":ri,"answer":t["answer"],"choice":choice,
          "correct":int(choice==t["answer"]),"confidence":float(prob[choice]),"margin":float(sc[order[0]]-sc[order[1]]),
          "entropy":float(-(prob*np.log(np.maximum(prob,1e-12))).sum()),"tokens":valid,"curvature":float(np.mean(cc)),
          "completion":texts[ri],"scores":sc.tolist()})
    del hs,seqs,masks
    if torch.cuda.is_available() and (ti+1)%5==0: torch.cuda.empty_cache()

F=pd.DataFrame(rows); X=np.stack(full_emb); Z=np.stack(proj_emb)
F.to_parquet(OUT/"traces.parquet")
np.savez_compressed(OUT/"emb_full.npz",x=X,layers=LAYERS)
np.savez_compressed(OUT/"emb_proj.npz",x=Z,layers=LAYERS)
summary={"run":RUN,"domain":DOMAIN,"tasks":N,"trajectories":len(F),"accuracy":float(F.correct.mean()),
 "curvature_success":float(F[F.correct==1].curvature.mean()),"curvature_failure":float(F[F.correct==0].curvature.mean()),
 "elapsed":time.time()-start,"gpu":torch.cuda.get_device_name(0),"layers":LAYERS}
(OUT/"shard_summary.json").write_text(json.dumps(summary,indent=2))
archive=OUT.parent/(RUN+".zip")
with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as zf:
    for p in OUT.iterdir(): zf.write(p,p.name)
url=upload(archive)
print("SHARD_SUMMARY="+json.dumps(summary),flush=True)
print("SHARD_URL="+url,flush=True)
print("SUMMARY_B64="+base64.b64encode((OUT/"shard_summary.json").read_bytes()).decode(),flush=True)
