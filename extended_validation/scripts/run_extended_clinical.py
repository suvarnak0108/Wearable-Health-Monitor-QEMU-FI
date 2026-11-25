"""Additive MIT-BIH validation. Run from repository root in existing venv."""
import ast, hashlib, json, os, re, shutil, subprocess, tempfile, time
from pathlib import Path
import numpy as np
import scipy
from scipy.signal import resample
import wfdb
ROOT=Path.cwd()
OUT=ROOT/'extended_validation/independent_clinical'
RECORDS='109 111 112 114 116 117 118 121 122 123 124 201 202 203 205 207 208 209 210 212'.split()
SCENARIOS=[('clean',0,False,False),('burst',0,True,False),('em_drift',.20,False,True),('triple',.15,True,True)]
EXPECTED='7ed24171dbf016ec1ce0b52cf1c74adf36b869cfd26265f6d78c3edeb7a69e04'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):
    t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(obj,indent=2,allow_nan=False)); t.replace(p)
def main():
    assert sha(ROOT/'firmware/firmware_m4.elf')==EXPECTED
    assert re.search(r'#define\s+REFRACTORY_SAMPLES\s+30U\b',(ROOT/'firmware/src/r_peak.c').read_text())
    OUT.mkdir(parents=True,exist_ok=True)
    for d in ('inputs','logs','trials'): (OUT/d).mkdir(exist_ok=True)
    u=ROOT/'simulation/run_unified_master_pipeline.py'
    paths=[u,Path(__file__).resolve(),ROOT/'data_cache/nstdb_em_dat.npz',*sorted(p for p in (ROOT/'firmware').rglob('*') if p.is_file() and p.suffix in ('.c','.h','.s','.ld') )]
    manifest=dict(records=RECORDS,scenarios=SCENARIOS,source_hashes={str(p):sha(p) for p in paths},numpy=np.__version__,scipy=scipy.__version__,wfdb=wfdb.__version__,tools={k:subprocess.check_output([k,'--version'],text=True).splitlines()[0] for k in ('arm-none-eabi-gcc','qemu-system-arm')})
    mp=OUT/'manifest.json'; manifest=json.loads(json.dumps(manifest))
    if mp.exists() and json.loads(mp.read_text())!=manifest: raise RuntimeError('Manifest changed; preserve results and investigate')
    save(mp,manifest)
    with np.load(ROOT/'data_cache/nstdb_em_dat.npz') as d: em=resample(d['sig'],6000)
    em/=np.std(em)
    rows=[]
    with tempfile.TemporaryDirectory(prefix='extended_qemu_') as tmp:
        p=Path(tmp); shutil.copytree(ROOT/'firmware',p/'firmware')
        env=dict(np=np,subprocess=subprocess,HEADER=p/'firmware/include/qemu_dataset_independent.h',ELF=p/'firmware/independent_trial.elf',C_SRCS=[str(p/'firmware/src'/n) for n in ('main.c','ecg_filter.c','r_peak.c','hr_estimator.c','fault_detector.c')])
        names={'compile_arm','write_header','make_realistic_aux','aami_score'}
        nodes=[n for n in ast.parse(u.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
        assert {n.name for n in nodes}==names
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(u),'exec'),env)
        for i,rec in enumerate(RECORDS):
            cache=OUT/'inputs'/f'{rec}.npz'
            if not cache.exists():
                for attempt in range(6):
                    try:
                        r=wfdb.rdrecord(rec,pn_dir='mitdb/1.0.0',sampto=21600); a=wfdb.rdann(rec,'atr',pn_dir='mitdb/1.0.0',sampto=21600)
                        assert r.fs==360 and len(r.p_signal)==21600
                        np.savez(cache,sig=r.p_signal[:,0],sample=a.sample,symbol=np.array(a.symbol)); break
                    except Exception:
                        if attempt==5: raise
                        time.sleep(3*(attempt+1))
            with np.load(cache) as d: sig,s,b=d['sig'],d['sample'],d['symbol']
            assert len(sig)==21600 and np.isfinite(sig).all()
            x=resample(sig,6000); x-=np.median(x); scale=np.percentile(abs(x),99); assert scale>0; x/=scale
            pk=np.array([round(int(j)*100/360) for j,k in zip(s,b) if k in {'N','L','R','A','V','F','e','j','E','J'}],dtype=int); pk=pk[(pk>0)&(pk<6000)]
            assert len(pk)>0
            windows=[pk[(pk>=w*1000)&(pk<(w+1)*1000)] for w in range(6)]
            ref=np.array([6000/np.median(np.diff(v)) if len(v)>=2 else 75. for v in windows]); assert np.isfinite(ref).all()
            for name,amp,burst,drift in SCENARIOS:
                key=f'{rec}_{name}'; dest=OUT/'trials'/f'{key}.json'
                if dest.exists():
                    row=json.loads(dest.read_text()); assert row['cache_sha256']==sha(cache); rows.append(row); continue
                a,o=env['make_realistic_aux'](np.repeat(ref,1000),900+i); dt=np.full(6000,10); gt=np.zeros(6000,dtype=np.uint8)
                if burst:
                    for j in range(300,4801,500): dt[j:j+10]=60; gt[j:j+10]=1
                if drift: a[2000:3500]+=22; gt[2000:3500]=1
                try:
                    os.chdir(p); env['write_header'](x+amp*np.roll(em,i*400),a,o,dt,gt); env['compile_arm']()
                    proc=subprocess.run(['qemu-system-arm','-M','mps2-an386','-cpu','cortex-m4','-kernel',str(env['ELF']),'-nographic','-semihosting-config','enable=on,target=native'],capture_output=True,text=True,timeout=30)
                    out=proc.stdout+proc.stderr; (OUT/'logs'/f'{key}.txt').write_text(out); proc.check_returncode()
                finally: os.chdir(ROOT)
                def grab(label):
                    m=re.search(re.escape(label)+r'\s*(\d+)',out)
                    if not m: raise RuntimeError('Missing '+label)
                    return int(m.group(1))
                tp,fp,fn,tn=[grab(k+':') for k in ('TP','FP','FN','TN')]; n=int(gt.sum()); assert tp+fn==n and fp+tn==6000-n
                count=grab('Detected R-peaks:'); m=re.search(r'PEAKS_CSV:[ \t]*([0-9,]*)',out); assert m is not None
                det=np.array([int(v) for v in m.group(1).split(',') if v],dtype=int); assert len(det)==count and np.all(np.diff(det)>0) and np.all((det>=0)&(det<6000))
                se,ppv,f1=env['aami_score'](pk,det)
                row=dict(record=rec,scenario=name,seed=900+i,cache_sha256=sha(cache),input_sha256=sha(env['HEADER']),elf_sha256=sha(env['ELF']),tp=tp,fp=fp,fn=fn,tn=tn,fault_precision=100*tp/(tp+fp) if n and tp+fp else None,fault_recall=100*tp/n if n else None,fault_f1=200*tp/(2*tp+fp+fn) if n else None,fpr=100*fp/(fp+tn),beat_sensitivity=se,beat_ppv=ppv,beat_f1=f1,reference_peaks=pk.tolist(),detected_peaks=det.tolist(),reference_hr=ref.tolist(),reference_fallback_windows=sum(len(v)<2 for v in windows),excluded_annotation_symbols=sorted(set(b)-{'N','L','R','A','V','F','e','j','E','J'}),sqi_below50_windows=grab('HoldWindows:'),stack_hwm=grab('StackHWM:'))
                for k,label in [('raw','HR_RAW'),('rr','HR_RR'),('hold','HR_MIT'),('sqi','SQI')]:
                    v=np.array([int(z)/100 for z in re.findall(label+r'\s+x100 = (-?\d+)',out)]); assert len(v)==6
                    row[k+'_windows']=v.tolist(); row[k+('_mean' if k=='sqi' else '_mae')]=float(np.mean(v if k=='sqi' else abs(v-ref)))
                save(dest,row); rows.append(row); print(f'{len(rows)}/80 {key}: beat F1={f1:.2f}, RR MAE={row["rr_mae"]:.3f}',flush=True)
    assert len(rows)==80 and sha(ROOT/'firmware/firmware_m4.elf')==EXPECTED
    assert all(sha(Path(k))==v for k,v in manifest['source_hashes'].items())
    save(OUT/'all_trials.json',rows)
    summary={}
    for name,*_ in SCENARIOS:
        group=[r for r in rows if r['scenario']==name]; summary[name]={}
        for k in ('beat_sensitivity','beat_ppv','beat_f1','fault_precision','fault_recall','fault_f1','fpr','raw_mae','rr_mae','hold_mae','sqi_mean','sqi_below50_windows'):
            v=np.array([r[k] for r in group if r[k] is not None]); summary[name][k]=dict(n=len(v),mean=float(v.mean()),sd=float(v.std(ddof=1))) if len(v)>1 else None
    save(OUT/'summary.json',summary); print(json.dumps(summary,indent=2)); print('COMPLETE: 80 trials')
if __name__=='__main__': main()
