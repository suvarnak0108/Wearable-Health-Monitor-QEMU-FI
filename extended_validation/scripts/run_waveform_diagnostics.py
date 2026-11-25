"""Eight isolated diagnostic executions; run from repository root."""
import ast, hashlib, json, os, re, shutil, subprocess, tempfile
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.signal import resample
import wfdb
ROOT=Path.cwd(); BASE=ROOT/'extended_validation/independent_clinical'; OUT=ROOT/'extended_validation/record108_diagnostics'
EXPECTED='7ed24171dbf016ec1ce0b52cf1c74adf36b869cfd26265f6d78c3edeb7a69e04'
LEGACY=set('N L R A V F e j E J'.split()); ALL=LEGACY|set('a S Q / f B n r ?'.split())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False))
def main():
    manifest=json.loads((BASE/'manifest.json').read_text()); assert all(sha(Path(k))==v for k,v in manifest['source_hashes'].items()); assert sha(ROOT/'firmware/firmware_m4.elf')==EXPECTED
    OUT.mkdir(parents=True,exist_ok=True)
    with np.load(ROOT/'data_cache/nstdb_em_dat.npz') as d: em=resample(d['sig'],6000)
    em/=np.std(em); rows=[]; provenance=[]
    for rec in manifest['records']:
        cache=BASE/'inputs'/f'{rec}.npz'
        with np.load(cache) as d: sig,s,b=d['sig'],d['sample'],d['symbol']
        r=wfdb.rdrecord(rec,pn_dir='mitdb/1.0.0',sampto=21600); a=wfdb.rdann(rec,'atr',pn_dir='mitdb/1.0.0',sampto=21600)
        ok=bool(np.array_equal(sig,r.p_signal[:,0]) and np.array_equal(s,a.sample) and np.array_equal(b,a.symbol)); provenance.append(dict(record=rec,exact_match=ok,cache_sha256=sha(cache),symbols=dict(Counter(b.tolist())))); save(OUT/'provenance.json',provenance)
        if not ok: raise RuntimeError('PhysioNet mismatch '+rec)
        print('Verified',rec,flush=True)
    u=ROOT/'simulation/run_unified_master_pipeline.py'
    with tempfile.TemporaryDirectory(prefix='waveform_diag_') as tmp:
        p=Path(tmp); shutil.copytree(ROOT/'firmware',p/'firmware'); rp=p/'firmware/src/r_peak.c'; original=rp.read_text()
        env=dict(np=np,subprocess=subprocess,HEADER=p/'firmware/include/qemu_dataset_independent.h',ELF=p/'firmware/independent_trial.elf',C_SRCS=[str(p/'firmware/src'/n) for n in ('main.c','ecg_filter.c','r_peak.c','hr_estimator.c','fault_detector.c')]); names={'compile_arm','write_header','make_realistic_aux','aami_score'}; nodes=[n for n in ast.parse(u.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]; assert {n.name for n in nodes}==names; exec(compile(ast.Module(body=nodes,type_ignores=[]),str(u),'exec'),env)
        for rec,ms,scenario in [('108',m,s) for m in (300,480) for s in ('clean','triple')]+[('114',300,s) for s in ('clean','em_drift')]+[('207',300,s) for s in ('clean','triple')]:
            if rec=='108':
                i=5
                with np.load(ROOT/'data_cache/mitdb_108_dat.npz',allow_pickle=True) as d: sig=d['sig']
                with np.load(ROOT/'data_cache/mitdb_108_atr.npz',allow_pickle=True) as d: s,b=d['sample'],d['symbol']
            else:
                i=manifest['records'].index(rec)
                with np.load(BASE/'inputs'/f'{rec}.npz') as d: sig,s,b=d['sig'],d['sample'],d['symbol']
            x=resample(sig,6000); x-=np.median(x); x/=np.percentile(abs(x),99)
            def peaks(symbols):
                v=np.array([round(int(j)*100/360) for j,k in zip(s,b) if k in symbols],dtype=int); return v[(v>0)&(v<6000)]
            pk=peaks(LEGACY); broad=peaks(ALL)
            def refs(v):
                w=[v[(v>=j)&(v<j+1000)] for j in range(0,6000,1000)]; return [float(6000/np.median(np.diff(z))) if len(z)>=2 else None for z in w]
            legacy=refs(pk); ref=np.array([75 if v is None else v for v in legacy]); a,o=env['make_realistic_aux'](np.repeat(ref,1000),900+i); dt=np.full(6000,10); gt=np.zeros(6000,dtype=np.uint8); amp={'clean':0,'triple':.15,'em_drift':.20}[scenario]
            if scenario=='triple':
                for j in range(300,4801,500): dt[j:j+10]=60; gt[j:j+10]=1
            if scenario!='clean': a[2000:3500]+=22; gt[2000:3500]=1
            y=x+amp*np.roll(em,i*400); key=f'{rec}_{ms}_{scenario}'; text,n=re.subn(r'(#define\s+REFRACTORY_SAMPLES\s+)30U\b',lambda m:m[1]+str(ms//10)+'U',original); assert n==1; rp.write_text(text)
            try:
                os.chdir(p); env['write_header'](y,a,o,dt,gt); env['compile_arm'](); proc=subprocess.run(['qemu-system-arm','-M','mps2-an386','-cpu','cortex-m4','-kernel',str(env['ELF']),'-nographic','-semihosting-config','enable=on,target=native'],capture_output=True,text=True,timeout=30); out=proc.stdout+proc.stderr; (OUT/f'{key}.txt').write_text(out); proc.check_returncode()
            finally: os.chdir(ROOT)
            m=re.search(r'PEAKS_CSV:[ \t]*([0-9,]*)',out); assert m; det=np.array([int(v) for v in m[1].split(',') if v],dtype=int); assert len(det)==int(re.search(r'Detected R-peaks:\s*(\d+)',out)[1]); counts={k:int(re.search(k+r':\s*(\d+)',out)[1]) for k in ('TP','FP','FN','TN')}; assert counts['TP']+counts['FN']==gt.sum() and sum(counts.values())==6000
            row=dict(record=rec,refractory_ms=ms,scenario=scenario,counts=counts,reference_peaks=pk.tolist(),expanded_reference_peaks=broad.tolist(),detected_peaks=det.tolist(),legacy_reference=legacy,expanded_reference=refs(broad),beat_scores=env['aami_score'](pk,det),expanded_beat_scores=env['aami_score'](broad,det),annotation_symbols=dict(Counter(b.tolist())),elf_sha256=sha(env['ELF']),header_sha256=sha(env['HEADER']))
            for k,label in [('raw','HR_RAW'),('rr','HR_RR'),('hold','HR_MIT'),('sqi','SQI')]:
                v=np.array([int(z)/100 for z in re.findall(label+r'\s+x100 = (-?\d+)',out)]); assert len(v)==6; row[k]=v.tolist()
            np.savez_compressed(OUT/f'{key}.npz',clean_ecg=x,input_ecg=y,aux_hr=a,dt=dt,gt=gt,annotation_samples=s,annotation_symbols=b,detected_peaks=det,reference_peaks=pk)
            save(OUT/f'{key}.json',row); rows.append(row); print(key,row['beat_scores'],flush=True)
    assert sha(ROOT/'firmware/firmware_m4.elf')==EXPECTED; assert all(sha(Path(k))==v for k,v in manifest['source_hashes'].items()); save(OUT/'diagnostics.json',rows); print('COMPLETE: 8 diagnostic trials')
if __name__=='__main__': main()
