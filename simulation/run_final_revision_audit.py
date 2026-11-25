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
def stale():
    names=['synth_gaussian_ecg','hr_trajectory_to_beats','make_1hz_aux','biquad_bandpass_3stage','detect_rpeaks_adaptive','rolling_9beat_hr','classify_faults_5stage']
    h=defs(ROOT/'simulation/resolve_remaining_audit_points.py',names,dict(np=np,butter=butter,sosfilt=sosfilt))
    hr=np.repeat([70.,84.,75.,95.,80.,70.],1000); x,_=h['hr_trajectory_to_beats'](hr)
    def roll(x): return h['rolling_9beat_hr'](h['detect_rpeaks_adaptive'](h['biquad_bandpass_3stage'](x)),6000)
    clean=roll(x); starts=[s for s in range(200,5635) if not any(b<=s<b+200 for b in range(1000,6000,1000))]
    groups={}
    for tr in range(20):
        seed=2000+tr; s=int(np.random.default_rng(seed).choice(starts)); e=s+165; a,o=h['make_1hz_aux'](hr,seed)
        for compound in (False,True):
            dt=np.full(6000,10); gt=np.zeros(6000,dtype=bool); y=x.copy(); y[s:e]=y[s-1]; gt[s:e]=True
            if compound:
                for b in range(5):
                    j=400+b*900; dt[j:j+10]=60; gt[j:j+10]=True
            for mode,r in [('legacy_clean_hr',clean),('corrupted_hr',roll(y))]:
                c=h['classify_faults_5stage'](dt,y,a,o,r); c=c[0] if isinstance(c,tuple) else c; row=metrics(c!=0,gt); row.update(seed=seed,start=s,stale_recall=float(np.mean(c[s:e]==4)*100)); groups.setdefault(('compound' if compound else 'f5')+'_'+mode,[]).append(row)
    save('stale_confusion_matrices',{k:dict(summary=summary(v),trials=v) for k,v in groups.items()})
def qemu_audit():
    import wfdb
    with tempfile.TemporaryDirectory(prefix='wearable_audit_') as tmp:
        p=Path(tmp); shutil.copytree(ROOT/'firmware',p/'firmware');
        env=dict(np=np,subprocess=subprocess,re=re,butter=butter,lfilter=lfilter,resample=resample,ROOT=p,HEADER=p/'firmware/include/qemu_dataset_independent.h',ELF=p/'firmware/independent_trial.elf',C_SRCS=[str(p/'firmware/src'/n) for n in ['main.c','ecg_filter.c','r_peak.c','hr_estimator.c','fault_detector.c']])
        names=['compile_arm','write_header','run_qemu','make_step_ecg','make_realistic_aux','make_continuous_ecg','pt_causal_100hz','aami_score']
        defs(ROOT/'simulation/run_unified_master_pipeline.py',names,env)
        import os
        os.chdir(p)
        def run(x,a,o,dt,gt):
            z=env['run_qemu'](x,a,o,dt,gt.astype(np.uint8)); rp,pk,holds,tp,fp,fn,tn,hwm,raw,rr,mit,sqi=z
            if tp+fp+fn+tn!=6000 or any(len(v)!=6 for v in (raw,rr,mit,sqi)) or len(pk)!=rp: raise RuntimeError('Invalid QEMU output')
            return dict(peaks=rp,holds=holds,tp=tp,fp=fp,fn=fn,tn=tn,raw=raw,rr=rr,mit=mit,sqi=sqi,pk=pk,hwm=hwm)
        try:
            if not (OUT/'mains_phase_sensitivity.json').exists():
                x,hr,ref=env['make_step_ecg'](); t=np.arange(6000)/100; rows=[]
                for f,phase in [(50,v) for v in (0,30,45,60,90)]+[(49,0),(48,0)]:
                    n=.25*np.sin(2*np.pi*f*t+np.deg2rad(phase)); z=run(x+n,hr,np.full(6000,98.),np.full(6000,10),np.zeros(6000,dtype=bool)); rows.append(dict(frequency=f,phase_deg=phase,injected_rms=float(np.sqrt(np.mean(n*n))),injected_peak=float(np.max(abs(n))),peaks=z['peaks'],rr_mae=float(np.mean(abs(z['rr']-ref))),raw_mae=float(np.mean(abs(z['raw']-ref)))))
                save('mains_phase_sensitivity',rows)
            else:
                print('Skipping completed mains audit', flush=True)
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
                    z=run(x+amp*np.roll(em,i*400),a,o,dt,gt); tp,fp,fn,tn=[z[k] for k in ('tp','fp','fn','tn')]; rows.append(dict(record=rec,tp=tp,fp=fp,fn=fn,tn=tn,precision=100*tp/(tp+fp) if tp+fp and tp+fn else None,recall=100*tp/(tp+fn) if tp+fn else None,f1=200*tp/(2*tp+fp+fn) if tp+fn else None,fpr=100*fp/(fp+tn),raw_mae=float(np.mean(abs(z['raw']-ref))),rr_mae=float(np.mean(abs(z['rr']-ref))),hold_mae=float(np.mean(abs(z['mit']-ref))),sqi=float(np.mean(z['sqi'])),holds=z['holds']))
                groups[name]=dict(summary=summary(rows),records=rows); save('clinical_uncertainty',groups)
        finally: os.chdir(ROOT)
if __name__=='__main__':
    import scipy
    save('environment_versions',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'simulation/resolve_remaining_audit_points.py',ROOT/'simulation/run_unified_master_pipeline.py',*sorted((ROOT/'firmware/src').glob('*.c'))]}))
    if not (OUT/'stale_confusion_matrices.json').exists():
        stale()
    else:
        print('Skipping completed stale audit', flush=True)
    if '--host-only' not in sys.argv: qemu_audit()
    print('Audit outputs:',OUT)
