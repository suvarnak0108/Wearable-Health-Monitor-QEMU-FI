"""Run from repository root. 530 supplementary QEMU trials; resumable audit_v6."""
import ast, hashlib, json, os, re, shutil, subprocess, tempfile, sys
from pathlib import Path
import numpy as np
from scipy.signal import resample
ROOT=Path.cwd(); OUT=ROOT/'results/audit_v6'; OUT.mkdir(parents=True,exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    t=p.with_suffix('.tmp'); t.write_text(json.dumps(v,indent=2,allow_nan=False)); t.replace(p)
def defs(path,names,env):
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes}!=set(names): raise RuntimeError('Missing functions: '+str(path))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env); return env
U=ROOT/'simulation/run_unified_master_pipeline.py'; H=ROOT/'simulation/resolve_remaining_audit_points.py'
h=defs(H,['synth_gaussian_ecg','hr_trajectory_to_beats','make_1hz_aux','windowed_10s_hr_rr_filtered'],dict(np=np))
u=defs(U,['make_step_ecg','make_realistic_aux','make_continuous_ecg'],dict(np=np))
step,hr,ref=u['make_step_ecg'](); const=np.full(6000,80.); cx,cp=h['hr_trajectory_to_beats'](const); cref=h['windowed_10s_hr_rr_filtered'](cp)
noise={}
for k in ('bw','ma','em'):
    with np.load(ROOT/'data_cache'/f'nstdb_{k}_dat.npz') as d: v=resample(d['sig'],6000)
    noise[k]=v/np.std(v)
def cases():
    for name,k,amp in [('clean',None,0),('bw', 'bw',.25),('ma','ma',.20),('em25','em',.25),('em35','em',.35)]:
        for t in range(20):
            x,_,a,o,r,_=u['make_continuous_ecg'](500+t); dt=np.full(6000,10); g=np.zeros(6000,dtype=np.uint8); rng=np.random.default_rng(700+t)
            if k:
                x=x+amp*np.roll(noise[k],t*293); ix=rng.choice(np.arange(100,5900),150,replace=False); a[ix]+=rng.uniform(18,40,150); g[ix]=5
            yield 'nstdb',name,t,500+t,x,a,o,dt,g,r,dict(amplitude=amp,injection_seed=700+t)
    for name in ('missing','burst','range','stale','drift'):
        for t in range(20):
            x=step.copy(); a,o=u['make_realistic_aux'](hr,200+t); dt=np.full(6000,10); g=np.zeros(6000,dtype=np.uint8); rng=np.random.default_rng(300+t)
            ix=rng.choice(np.arange(100,5900),150,replace=False)
            if name=='missing': dt[ix]=25; g[ix]=1
            elif name=='burst':
                for j in range(200,5241,360): dt[j:j+10]=60; g[j:j+10]=2
            elif name=='range': a[ix]=245; g[ix]=3
            elif name=='stale':
                j=2000+(t*37)%1500; x[j:j+165]=x[j-1]; g[j:j+165]=4
            else: a[ix]+=22; g[ix]=5
            yield 'single',name,t,200+t,x,a,o,dt,g,ref,dict(injection_seed=300+t)
    for off in (10,11,12,13,14):
        for t in range(20):
            a,o=h['make_1hz_aux'](const,800+t); a[2000:3500]+=off; g=np.zeros(6000,dtype=np.uint8); g[2000:3500]=5
            yield 'boundary',str(off),t,800+t,cx,a,o,np.full(6000,10),g,cref,dict(offset_bpm=off)
    for name in ('missing','range','drift'):
        for density in (.01,.025,.05,.10,.20):
            for t in range(10):
                a,o=u['make_realistic_aux'](hr,850+t); dt=np.full(6000,10); g=np.zeros(6000,dtype=np.uint8); ix=np.random.default_rng(800+t).choice(np.arange(100,5900),round(6000*density),replace=False)
                if name=='missing': dt[ix]=25; g[ix]=1
                elif name=='range': a[ix]=245; g[ix]=3
                else: a[ix]+=22; g[ix]=5
                yield 'density',f'{name}_{density}',t,850+t,step,a,o,dt,g,ref,dict(density=density,injection_seed=800+t)
    for name in ('nm','bs','nd','tc'):
        for t in range(20):
            rng=np.random.default_rng(400+t); x=step.copy(); a,o=u['make_realistic_aux'](hr,450+t); dt=np.full(6000,10); g=np.zeros(6000,dtype=np.uint8)
            if name!='bs': x+=rng.normal(0,.15,6000)
            if name in ('nm','tc'):
                ix=rng.choice(np.arange(100,5900),100,replace=False); dt[ix]=25; g[ix]=1
            if name=='bs':
                for j in range(200,3081,320): dt[j:j+5]=60; g[j:j+5]=2
                j=3600+(t*29)%1200; x[j:j+165]=x[j-1]; g[j:j+165]=4
            if name in ('nd','tc'):
                ix=rng.choice(np.where(g==0)[0][10:-10],100,replace=False); a[ix]+=22; g[ix]=5
            yield 'compound',name,t,450+t,x,a,o,dt,g,ref,dict(injection_seed=400+t)

def main():
    manifest=dict(script=sha(Path(__file__)),sources={str(p.relative_to(ROOT)):sha(p) for p in [U,H,*sorted((ROOT/'firmware/src').glob('*.c')),ROOT/'firmware/include/wearable_monitor.h',ROOT/'firmware/startup/startup.s',ROOT/'firmware/linker/mps2_an386.ld',*sorted((ROOT/'data_cache').glob('nstdb_*_dat.npz'))]},numpy=np.__version__,tools={k:subprocess.check_output([k,'--version'],text=True).splitlines()[0] for k in ('arm-none-eabi-gcc','qemu-system-arm')})
    mp=OUT/'manifest.json'; rp=OUT/'trials.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest: raise RuntimeError('Resume manifest mismatch; preserve existing results')
    save(mp,manifest); rows=json.loads(rp.read_text()) if rp.exists() else []; done={r['id'] for r in rows}
    if len(done)!=len(rows): raise RuntimeError('Duplicate trial IDs')
    with tempfile.TemporaryDirectory(prefix='primary_qemu_') as tmp:
        p=Path(tmp); shutil.copytree(ROOT/'firmware',p/'firmware'); m=p/'firmware/src/main.c'; s=m.read_text()
        if not re.search(r'#define\s+REFRACTORY_SAMPLES\s+30U\b',(p/'firmware/src/r_peak.c').read_text()): raise RuntimeError('Expected original 300 ms refractory')
        # Add observation counters only. Keep binary ground truth for original classifier metrics.
        for old,new in [('uint32_t hold_engaged_windows = 0U;', 'uint32_t hold_engaged_windows = 0U;\n    uint32_t window_flags=0, stale_hits=0;'),('uint8_t gt = fault_ground_truth[n];','uint8_t gt = fault_ground_truth[n];\n            if (n>=2000U && n<3500U && fc==FAULT_HR_INCONSISTENCY) window_flags++;\n            if (fc==FAULT_STALE_DATA && audit_stale[n]) stale_hits++;'),('print_kv("TP: ", (int32_t)tp);','print_kv("AuditWindow: ", (int32_t)window_flags);\n    print_kv("AuditStale: ", (int32_t)stale_hits);\n    print_kv("TP: ", (int32_t)tp);')]:
            if s.count(old)!=1: raise RuntimeError('Unexpected main.c: '+old)
            s=s.replace(old,new)
        m.write_text(s)
        env=defs(U,['compile_arm','write_header'],dict(np=np,subprocess=subprocess,HEADER=p/'firmware/include/qemu_dataset_independent.h',ELF=p/'firmware/independent_trial.elf',C_SRCS=[str(p/'firmware/src'/n) for n in ('main.c','ecg_filter.c','r_peak.c','hr_estimator.c','fault_detector.c')]))
        os.chdir(p)
        try:
            for group,name,t,seed,x,a,o,dt,g,ref0,extra in cases():
                key=f'{group}/{name}/{t}'
                if key in done: continue
                env['write_header'](x,a,o,dt,(g!=0).astype(np.uint8)); header=env['HEADER']; text=header.read_text(); header.write_text(text.replace('#endif','static const uint8_t audit_stale[6000] = {'+','.join(map(str,(g==4).astype(int)))+'};\n#endif'))
                env['compile_arm'](); proc=subprocess.run(['qemu-system-arm','-M','mps2-an386','-cpu','cortex-m4','-kernel',str(env['ELF']),'-nographic','-semihosting-config','enable=on,target=native'],capture_output=True,text=True,timeout=30,check=True); out=proc.stdout+proc.stderr
                def grab(k):
                    z=re.search(re.escape(k)+r'\s*(\d+)',out)
                    if not z: raise RuntimeError('Missing '+k+' in '+out)
                    return int(z.group(1))
                tp,fp,fn,tn=[grab(k+':') for k in ('TP','FP','FN','TN')]; n=int(np.count_nonzero(g)); assert tp+fn==n and fp+tn==6000-n
                row=dict(id=key,group=group,scenario=name,trial=t,seed=seed,**extra,injected=n,tp=tp,fp=fp,fn=fn,tn=tn,precision=100*tp/(tp+fp) if n and tp+fp else (0. if n else None),recall=100*tp/n if n else None,f1=200*tp/(2*tp+fp+fn) if n else None,fpr=100*fp/(fp+tn),peaks=grab('Detected R-peaks:'),holds=grab('HoldWindows:'),instrumented_stack_hwm=grab('StackHWM:'),elf_sha256=sha(env['ELF']),input_sha256=sha(header))
                for k,label in [('raw','HR_RAW'),('rr','HR_RR'),('hold','HR_MIT'),('sqi','SQI')]:
                    v=np.array([int(i)/100 for i in re.findall(label+r'\s+x100 = (\d+)',out)]); assert len(v)==6; row[k+'_windows']=v.tolist(); row[k+'_mean' if k=='sqi' else k+'_mae']=float(np.mean(v if k=='sqi' else abs(v-ref0)))
                row['reference_hr']=ref0.tolist(); row['boundary_flag_pct']=100*grab('AuditWindow:')/1500 if group=='boundary' else None; row['stale_same_class_recall']=100*grab('AuditStale:')/int(np.sum(g==4)) if np.any(g==4) else None
                rows.append(row); save(rp,rows); print(f'{len(rows)}/530 {key}',flush=True)
        finally: os.chdir(ROOT)
    assert len(rows)==530
    groups={}
    for r in rows: groups.setdefault(r['group']+'/'+r['scenario'],[]).append(r)
    summary={}
    for k,rr in groups.items():
        summary[k]={'n':len(rr),'metrics':{}}
        for field in ('tp','fp','fn','tn','precision','recall','f1','fpr','peaks','holds','raw_mae','rr_mae','hold_mae','sqi_mean','boundary_flag_pct','stale_same_class_recall'):
            v=[r[field] for r in rr if r[field] is not None]
            if v: summary[k]['metrics'][field]=dict(mean=float(np.mean(v)),sd=float(np.std(v,ddof=1)),n=len(v))
    save(OUT/'summary.json',summary); print('COMPLETE: 530 trials',flush=True)
if __name__=='__main__': main()
