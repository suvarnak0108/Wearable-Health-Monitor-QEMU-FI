"""Supplementary audit; run from repository root. Never overwrites v3/v4 or firmware."""
import ast, json, shutil, tempfile, re, subprocess, sys, platform, hashlib
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfilt, lfilter, resample
ROOT=Path.cwd()
OUT=ROOT/'results/audit_v5'; OUT.mkdir(parents=True,exist_ok=True)
def save(name,obj):
    (OUT/(name+'.json')).write_text(json.dumps(obj,indent=2,allow_nan=False))
def defs(path,names,env):
    tree=ast.parse(path.read_text()); nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes}!=set(names): raise RuntimeError('Missing definitions in '+str(path))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    return env
def metrics(pred,gt):
    tp=int(np.sum(pred & gt)); fp=int(np.sum(pred & ~gt)); fn=int(np.sum(~pred & gt)); tn=int(np.sum(~pred & ~gt))
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=100*tp/(tp+fp) if tp+fp else None,recall=100*tp/(tp+fn) if tp+fn else None,f1=200*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,fpr=100*fp/(fp+tn) if fp+tn else None)
def summary(rows):
    return {k:dict(mean=float(np.mean(v)),sd=float(np.std(v,ddof=1)) if len(v)>1 else None,n=len(v)) for k in rows[0] if k not in ('seed','start','record') and (v:=[r[k] for r in rows if isinstance(r[k],(int,float))])}
def score(ref,det,delay):
    a=sorted(map(int,ref)); b=sorted(int(x)-delay for x in det)
    i=j=tp=0
    while i<len(a) and j<len(b):
        if b[j]<a[i]-15: j+=1
        elif b[j]>a[i]+15: i+=1
        else: tp+=1; i+=1; j+=1
    return 200*tp/(len(a)+len(b)) if len(a)+len(b) else None
def qemu_audit():
    import wfdb
    with tempfile.TemporaryDirectory(prefix='wearable_audit_') as tmp:
        p=Path(tmp); shutil.copytree(ROOT/'firmware',p/'firmware');
        main=p/'firmware/src/main.c'
        text=main.read_text()
        for old,new in [
            ('BiquadFilterState filt;', ''),
            ('RPeakDetectorState rdet;', ''),
            ('ECG_Filter_Init(&filt);', ''),
            ('RPeak_Init(&rdet);', ''),
            ('float y = ECG_Filter_Process(&filt, ecg_data[n]);', ''),
            ('RPeak_Process(&rdet, y, n)', 'PanTompkins_M4_Step(ecg_data[n], n)')]:
            if text.count(old)!=1: raise RuntimeError('Unexpected main source: '+old)
            text=text.replace(old,new)
        main.write_text('extern unsigned int PanTompkins_M4_Step(float, unsigned int);\n'+text)
        pt=p/'firmware/pt_m4_bench.c'
        text=pt.read_text()
        text,n=re.subn(r'int main\(void\)\s*\{[^}]*\}', '', text)
        if n!=1: raise RuntimeError('Unexpected PT benchmark entry point')
        pt.write_text(text)
        env=dict(np=np,subprocess=subprocess,re=re,butter=butter,lfilter=lfilter,resample=resample,ROOT=p,HEADER=p/'firmware/include/qemu_dataset_independent.h',ELF=p/'firmware/independent_trial.elf',C_SRCS=[str(p/'firmware/src'/n) for n in ['main.c','hr_estimator.c','fault_detector.c']]+[str(pt)])
        names=['compile_arm','write_header','run_qemu','make_step_ecg','make_realistic_aux','make_continuous_ecg','pt_causal_100hz','aami_score']
        defs(ROOT/'simulation/run_unified_master_pipeline.py',names,env)
        import os
        os.chdir(p)
        def run(x,a,o,dt,gt):
            z=env['run_qemu'](x,a,o,dt,gt.astype(np.uint8)); rp,pk,holds,tp,fp,fn,tn,hwm,raw,rr,mit,sqi=z
            if tp+fp+fn+tn!=6000 or any(len(v)!=6 for v in (raw,rr,mit,sqi)) or len(pk)!=rp: raise RuntimeError('Invalid QEMU output')
            return dict(peaks=rp,holds=holds,tp=tp,fp=fp,fn=fn,tn=tn,raw=raw,rr=rr,mit=mit,sqi=sqi,pk=pk,hwm=hwm)
        try:
            records=['100','101','103','105','106','108','113','115','119','200']; syms={'N','L','R','A','V','F','e','j','E','J'}
            def cached(rec,db,ann=False):
                f=ROOT/'data_cache'/f'{db}_{rec}_{"atr" if ann else "dat"}.npz'
                if f.exists():
                    with np.load(f,allow_pickle=True) as d: return (d['sample'],d['symbol']) if ann else d['sig']
                raise FileNotFoundError(f'Original cache required: {f}')
            em=resample(cached('em','nstdb'),6000); em/=np.std(em)
            groups={}
            for name,amp,burst,drift in [('clean',0,False,False),('burst',0,True,False),('em_drift',.20,False,True),('triple',.15,True,True)]:
                rows=[]
                for i,rec in enumerate(records):
                    x=resample(cached(rec,'mitdb'),6000); x=(x-np.median(x))/np.percentile(abs(x-np.median(x)),99); s,b=cached(rec,'mitdb',True); pk=np.array([round(int(j)*100/360) for j,k in zip(s,b) if k in syms]); pk=pk[(pk>0)&(pk<6000)]
                    ref=np.array([6000/np.median(np.diff(v)) if len(v)>=2 else 75. for w in range(6) for v in [pk[(pk>=w*1000)&(pk<(w+1)*1000)]]]); a,o=env['make_realistic_aux'](np.repeat(ref,1000),900+i); dt=np.full(6000,10); gt=np.zeros(6000,dtype=bool)
                    if burst:
                        for j in range(300,4801,500): dt[j:j+10]=60; gt[j:j+10]=True
                    if drift: a[2000:3500]+=22; gt[2000:3500]=True
                    z=run(x+amp*np.roll(em,i*400),a,o,dt,gt); tp,fp,fn,tn=[z[k] for k in ('tp','fp','fn','tn')]; rows.append(dict(record=rec,beat_f1_by_delay={str(d):score(pk,z['pk'],d) for d in (0,5,10,15,20)},peak_indices=list(map(int,z['pk'])),reference_indices=list(map(int,pk)),stack_hwm=z['hwm'],tp=tp,fp=fp,fn=fn,tn=tn,precision=(100*tp/(tp+fp) if tp+fp else 0.) if tp+fn else None,recall=100*tp/(tp+fn) if tp+fn else None,f1=200*tp/(2*tp+fp+fn) if tp+fn else None,fpr=100*fp/(fp+tn),raw_mae=float(np.mean(abs(z['raw']-ref))),rr_mae=float(np.mean(abs(z['rr']-ref))),hold_mae=float(np.mean(abs(z['mit']-ref))),sqi=float(np.mean(z['sqi'])),holds=z['holds']))
                groups[name]=dict(summary=summary(rows),records=rows); save('pt_style_clinical',groups); print('PT-style',name,flush=True)
            save('pt_style_memory', {cmd:subprocess.check_output([cmd,*args,str(env['ELF'])],text=True) for cmd,args in [('arm-none-eabi-size',['-A']),('arm-none-eabi-nm',['-S','--size-sort'])]})
        finally: os.chdir(ROOT)

if __name__=='__main__':
    qemu_audit()
