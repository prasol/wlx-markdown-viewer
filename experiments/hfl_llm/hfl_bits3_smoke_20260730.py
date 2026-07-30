from __future__ import annotations
import gc, itertools, json, os, random, traceback
from dataclasses import dataclass
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

MARK='[[HFL_STATE_CHECKPOINT]]'; SEED=20260730
LAYER_FRACS=[.12,.25,.38,.50,.62,.75,.88]
ALPHAS=[.5,1.0]; WIDTHS=[1,4]
@dataclass
class Ex: id:str; prompt:str; answer:str; gid:str; role:str
@dataclass
class Group: gid:str; source:Ex; target:Ex; negative:Ex; qualification:Ex; probes:dict
@dataclass
class Enc: ids:list[int]; prompt_ids:list[int]; answer_ids:list[int]; marker:int

def exact_chain(rng,n,z):
    while True:
        s=[rng.randrange(2) for _ in range(3)]
        req=[q for q in range(3) if s[q]!=z[q]]
        if len(req)<=n and (n-len(req))%2==0: break
    ops=req[:]
    for _ in range((n-len(req))//2):
        q=rng.randrange(3); ops += [q,q]
    rng.shuffle(ops); x=s[:]
    for q in ops:x[q]^=1
    assert x==z and len(ops)==n
    return s,ops

def parity(z,mask): return sum(z[q] for q in range(3) if mask>>q&1)%2

def prompt(s,ops,mask,probe=False):
    lines=[f'Track three binary registers A={s[0]}, B={s[1]}, C={s[2]}.',
           'Apply every flip exactly in order. Silently consolidate the final values at the checkpoint.']
    for i,q in enumerate(ops,1): lines.append(f'{i}. Flip '+['A.','B.','C.'][q])
    lines += ['State checkpoint: '+MARK]
    if probe: lines += ['Continue from this consolidated checkpoint state carefully.']
    else:
        names=[['A','B','C'][q] for q in range(3) if mask>>q&1]
        lines += ['Output the XOR parity of '+' and '.join(names)+'.','Reply with exactly one character: 0 or 1.']
    return '\n'.join(lines)

def make_groups(n=40):
    rng=random.Random(SEED); out=[]
    for gi in range(n):
        z=[rng.randrange(2) for _ in range(3)]
        zp=z[:]
        while zp==z: zp=[rng.randrange(2) for _ in range(3)]
        ns=rng.choice([3,4,5]); nt=rng.choice([10,11,12,13])
        ss,so=exact_chain(rng,ns,z); ts,to=exact_chain(rng,nt,z); nn,no=exact_chain(rng,nt,zp)
        mt=rng.choice([1,2,3,4,5,6,7]); ms=rng.choice([1,2,3,4,5,6,7])
        at=parity(z,mt); an=parity(zp,mt); a_s=parity(z,ms)
        tries=0
        while (ms==mt or a_s==at or an==at) and tries<100:
            if an==at:
                zp=z[:]; q=rng.randrange(3); zp[q]^=1
                nn,no=exact_chain(rng,nt,zp); an=parity(zp,mt)
            ms=rng.choice([1,2,3,4,5,6,7]); a_s=parity(z,ms); tries+=1
        if ms==mt or a_s==at or an==at: continue
        gid=f'b3_{gi:03d}'
        src=Ex(gid+':source',prompt(ss,so,ms),str(a_s),gid,'source')
        tgt=Ex(gid+':target',prompt(ts,to,mt),str(at),gid,'target')
        neg=Ex(gid+':negative',prompt(nn,no,mt),str(an),gid,'negative')
        qual=Ex(gid+':qualification',prompt(ss,so,mt),str(at),gid,'qualification')
        probes={'source':prompt(ss,so,ms,True),'target':prompt(ts,to,mt,True),'negative':prompt(nn,no,mt,True)}
        out.append(Group(gid,src,tgt,neg,qual,probes))
    return out

def flat(x):
    if hasattr(x,'input_ids'):x=x.input_ids
    if isinstance(x,dict):x=x['input_ids']
    if hasattr(x,'tolist'):x=x.tolist()
    if x and isinstance(x[0],list):x=x[0]
    return [int(v) for v in x]

def find_marker(tok,ids):
    ends=set()
    for text in [MARK,' '+MARK,MARK+' ',' '+MARK+' ']:
        nd=flat(tok.encode(text,add_special_tokens=False));n=len(nd)
        for i in range(len(ids)-n+1):
            if ids[i:i+n]==nd:ends.add(i+n-1)
    if len(ends)==1:return next(iter(ends))
    dec=tok.decode(ids,skip_special_tokens=False)
    if dec.count(MARK)!=1:raise RuntimeError(f'marker ambiguous {ends}')
    for i in range(len(ids)):
        if MARK in tok.decode(ids[:i+1],skip_special_tokens=False):return i
    raise RuntimeError('marker absent')

def encode(tok,p,answer='0'):
    a=flat(tok.apply_chat_template([{'role':'user','content':p}],tokenize=True,add_generation_prompt=True,return_tensors='pt'))
    b=flat(tok.apply_chat_template([{'role':'user','content':p},{'role':'assistant','content':answer}],tokenize=True,add_generation_prompt=False,continue_final_message=True,return_tensors='pt'))
    if b[:len(a)]!=a:raise RuntimeError('assistant boundary mismatch')
    return Enc(b,a,b[len(a):],find_marker(tok,a))

def layers_of(m):
    for f in [lambda x:x.model.layers,lambda x:x.model.model.layers]:
        try:
            y=f(m)
            if len(y):return y
        except:pass
    raise RuntimeError('layers absent')
def score(m,e,dev):
    x=torch.tensor([e.ids],device=dev)
    with torch.inference_mode():lg=m(x,use_cache=False).logits[0]
    st=len(e.prompt_ids);v=[]
    for i,t in enumerate(e.answer_ids):v.append(torch.log_softmax(lg[st+i-1].float(),-1)[t].item())
    return float(np.mean(v))
def layer_ids(n):return sorted({min(n-1,max(0,round(f*(n-1)))) for f in LAYER_FRACS})
def capture(m,layers,lids,encs,dev,positions=None,width=4):
    out={}
    for key,e in encs.items():
        end=e.marker if positions is None else positions[key]; start=max(0,end-width+1); got={};hs=[]
        for li in lids:
            def mk(k):
                def hook(_mod,args):got[k]=args[0][0,start:end+1].detach().float().cpu()
                return hook
            hs.append(layers[li].register_forward_pre_hook(mk(li)))
        try:
            with torch.inference_mode():m(torch.tensor([e.prompt_ids],device=dev),use_cache=False)
        finally:
            for h in hs:h.remove()
        out[key]=got
    return out
def patch_score(m,layers,li,e,patch,dev,alpha,width):
    end=e.marker;start=max(0,end-width+1);p0=patch[-(end-start+1):]
    def pre(_mod,args):
        h=args[0].clone();p=p0.to(h.device,h.dtype);t=h[0,start:end+1]
        scale=t.float().norm(dim=-1,keepdim=True)/(p.float().norm(dim=-1,keepdim=True)+1e-8);p=p*scale.to(p.dtype)
        h[0,start:end+1]=(1-alpha)*t+alpha*p;return (h,)+tuple(args[1:])
    hk=layers[li].register_forward_pre_hook(pre)
    try:return score(m,e,dev)
    finally:hk.remove()
def cos(a,b):
    a=a.mean(0).numpy();b=b.mean(0).numpy();return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
def auc(y,s):
    try:return float(roc_auc_score(y,s))
    except:return .5
def maxstat(pos,neg):
    n,L=pos.shape;y=np.r_[np.ones(n),np.zeros(n)];obs=max(auc(y,np.r_[pos[:,j],neg[:,j]]) for j in range(L));ex=0
    for bits in itertools.product([0,1],repeat=n):
        mask=np.array(bits,bool);p=pos.copy();q=neg.copy();p[mask],q[mask]=neg[mask],pos[mask]
        ex+=max(auc(y,np.r_[p[:,j],q[:,j]]) for j in range(L))>=obs-1e-12
    return obs,(ex+1)/(2**n+1)
def boot(x,B=3000):
    x=np.array(x);r=np.random.default_rng(9);m=[r.choice(x,len(x),replace=True).mean() for _ in range(B)];return [float(np.quantile(m,.025)),float(np.quantile(m,.975))]
def load(mid):
    tok=AutoTokenizer.from_pretrained(mid,token=os.environ.get('HF_TOKEN'));tok.pad_token=tok.pad_token or tok.eos_token
    m=AutoModelForCausalLM.from_pretrained(mid,token=os.environ.get('HF_TOKEN'),device_map='auto',dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='sdpa');m.eval();m.config.use_cache=False
    return tok,m,layers_of(m),m.get_input_embeddings().weight.device

def run(mid,groups,fixed=None):
    print('MODEL_START',mid,flush=True);tok,m,layers,dev=load(mid);lids=layer_ids(len(layers));cand=groups if fixed is None else [g for g in groups if g.gid in fixed]
    base={};allenc={}
    for i,g in enumerate(cand):
        for e in [g.source,g.target,g.negative,g.qualification]:allenc[e.id]=encode(tok,e.prompt,e.answer);base[e.id]=score(m,allenc[e.id],dev)
        if (i+1)%8==0:print('BASELINE',i+1,len(cand),flush=True)
    rank=[]
    for g in cand:
        gap=base[g.qualification.id]-base[g.target.id];ok=base[g.qualification.id]>=-6 and gap>=.10;rank.append((ok,gap,g))
    rank.sort(key=lambda x:(x[0],x[1]),reverse=True);sel=[x[2] for x in rank[:8]] if fixed is None else cand;strict=sum(x[0] for x in rank[:8]) if fixed is None else sum(base[g.qualification.id]>=-6 and base[g.qualification.id]-base[g.target.id]>=.10 for g in sel)
    print('SELECTED',json.dumps([g.gid for g in sel]),'STRICT',strict,flush=True)
    enc={}
    for g in sel:
        for e in [g.source,g.target,g.negative]:enc[e.id]=allenc[e.id]
    states=capture(m,layers,lids,enc,dev,width=max(WIDTHS));probes={};pp={}
    for g in sel:
        for role,e in [('source',g.source),('target',g.target),('negative',g.negative)]:
            q=encode(tok,g.probes[role],'0');f=enc[e.id]
            if f.prompt_ids[:f.marker+1]!=q.prompt_ids[:q.marker+1]:raise RuntimeError('prefix mismatch')
            probes[e.id]=q;pp[e.id]=len(q.prompt_ids)-1
    future=capture(m,layers,lids,probes,dev,positions=pp,width=4);pos=np.zeros((8,len(lids)));neg=np.zeros_like(pos);curp=np.zeros_like(pos);curn=np.zeros_like(pos)
    for i,g in enumerate(sel):
        for j,li in enumerate(lids):
            pos[i,j]=cos(future[g.source.id][li],future[g.target.id][li]);neg[i,j]=cos(future[g.source.id][li],future[g.negative.id][li]);curp[i,j]=cos(states[g.source.id][li],states[g.target.id][li]);curn[i,j]=cos(states[g.source.id][li],states[g.negative.id][li])
    fa,pv=maxstat(pos,neg);ca,_=maxstat(curp,curn)
    combos=[(li,a,w) for li in lids for a in ALPHAS for w in WIDTHS];eff=np.zeros((8,len(combos)));rev=np.zeros_like(eff);selferr=[]
    for i,g in enumerate(sel):
        rg=sel[(i+1)%8];bt=base[g.target.id];bs=base[g.source.id]
        for c,(li,a,w) in enumerate(combos):
            mat=patch_score(m,layers,li,enc[g.target.id],states[g.source.id][li],dev,a,w)-bt
            wrong=patch_score(m,layers,li,enc[g.target.id],states[g.negative.id][li],dev,a,w)-bt
            rnd=patch_score(m,layers,li,enc[g.target.id],states[rg.source.id][li],dev,a,w)-bt
            eff[i,c]=mat-max(wrong,rnd)
            rev[i,c]=patch_score(m,layers,li,enc[g.source.id],states[g.target.id][li],dev,a,w)-bs
        selferr.append(abs(patch_score(m,layers,lids[len(lids)//2],enc[g.target.id],states[g.target.id][lids[len(lids)//2]],dev,1.0,4)-bt));print('PATCH',i+1,8,flush=True)
    oof=[];direct=[];chosen=[]
    for fold in range(4):
        te=np.array([i for i in range(8) if i%4==fold]);tr=np.array([i for i in range(8) if i%4!=fold]);c=int(np.argmax(eff[tr].mean(0)));chosen.append(combos[c])
        for i in te:oof.append(eff[i,c]);direct.append(eff[i,c]-rev[i,c])
    ci=boot(oof);summary={'model_id':mid,'selected_ids':[g.gid for g in sel],'strict_pairs':int(strict),'current_auc':ca,'future_auc':fa,'future_p':pv,'oof_mean':float(np.mean(oof)),'oof_ci':ci,'win_rate':float(np.mean(np.array(oof)>0)),'directionality':float(np.mean(direct)),'self_patch_error':float(max(selferr)),'chosen':chosen}
    summary['checks']={'pairs':strict>=8,'future_auc':fa>=.68,'future_p':pv<=.10,'oof':summary['oof_mean']>=.01,'ci':ci[0]>=-.02,'win':summary['win_rate']>=.58,'direction':summary['directionality']>=0,'self':summary['self_patch_error']<=.002};summary['pass']=all(summary['checks'].values());print('SUMMARY_JSON',json.dumps(summary,sort_keys=True),flush=True)
    del m;gc.collect();torch.cuda.empty_cache();return summary

def main():
    groups=make_groups();ll=run('unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit',groups);res={'llama':ll,'gemma':None}
    if ll['pass']:
        print('LLAMA_PASS_START_GEMMA',flush=True);res['gemma']=run('unsloth/gemma-2-9b-it-bnb-4bit',groups,set(ll['selected_ids']))
    else:print('LLAMA_FAIL_GEMMA_SKIPPED',flush=True)
    res['cascade_pass']=bool(res['gemma'] and res['gemma']['pass']);print('CASCADE_JSON',json.dumps(res,sort_keys=True),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();raise
