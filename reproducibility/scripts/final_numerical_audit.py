import argparse,json,statistics as st,math,hashlib
from pathlib import Path
from collections import Counter,defaultdict
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path('results'));p.add_argument('--out',type=Path,default=Path('final_numerical_results.json'));a=p.parse_args()
def load(v,n):return json.loads((a.results/f'audit_v{v}'/(n+'.json')).read_text())
def stats(rows):
 out={}
 for k in sorted(set().union(*(r.keys() for r in rows))):
  v=[r[k] for r in rows if isinstance(r.get(k),(int,float)) and k not in ('seed','trial')];
  if v:out[k]={'n':len(v),'mean':st.mean(v),'sd':st.stdev(v) if len(v)>1 else None}
 return out
def eq(x,y):assert math.isclose(x,y,rel_tol=1e-9,abs_tol=1e-9),(x,y)
def check(r):
 t,f,n,z=[r[k] for k in ('tp','fp','fn','tn')];assert min(t,f,n,z)>=0 and t+f+n+z==6000
 if 'injected' in r:assert t+n==r['injected']
 eq(r['fpr'],100*f/(f+z))
 if t+n:
  for k,v in [('precision',100*t/(t+f) if t+f else 0),('recall',100*t/(t+n)),('f1',200*t/(2*t+f+n))]:eq(r[k],v)
 else:assert all(r[k] is None for k in ('precision','recall','f1'))
 if 'reference_hr' in r:
  for k in ('raw','rr','hold'):eq(r[k+'_mae'],st.mean(abs(x-y) for x,y in zip(r[k+'_windows'],r['reference_hr'])))
r=load(6,'trials');assert len(r)==len({x['id'] for x in r})==530;assert len({x['input_sha256'] for x in r})==530
assert Counter(x['group'] for x in r)==dict(nstdb=100,single=100,boundary=100,density=150,compound=80)
g=defaultdict(list)
for x in r:check(x);g[x['group']+'/'+x['scenario']].append(x)
s={k:stats(v) for k,v in g.items()}; old=load(6,'summary');assert set(s)==set(old)
count=0
for k,v in old.items():
 assert v['n']==len(g[k])
 for m,w in v['metrics'].items():
  assert w['n']==s[k][m]['n'];eq(w['mean'],s[k][m]['mean']);eq(w['sd'],s[k][m]['sd']);count+=1
out={'audit_v6':s,'audit_v6_summary_metrics_checked':count,'audit_v5':{},'source_sha256':{}}
for name in ('clinical_uncertainty','pt_style_clinical','refractory_30_clinical','refractory_48_clinical'):
 d=load(5,name);out['audit_v5'][name]={}
 for k,v in d.items():
  rows=v['records'];assert len(rows)==len({x['record'] for x in rows})==10
  for x in rows:check(x)
  z=stats(rows);z['held_windows_total']=sum(x['holds'] for x in rows)
  if name=='pt_style_clinical':z['beat_f1_by_delay_samples']={delay:stats([{'f1':x['beat_f1_by_delay'][delay]} for x in rows])['f1'] for delay in ('0','5','10','15','20')}
  out['audit_v5'][name][k]=z
for name in ('refractory_30_noise','refractory_48_noise'):
 rows=load(5,name);assert len(rows)==100;out['audit_v5'][name]={str(k):stats([x for x in rows if x['sigma']==k]) for k in (0,.08,.15,.25,.35)}
for v in (5,6):
 for f in sorted((a.results/f'audit_v{v}').glob('*.json')):out['source_sha256'][str(f.relative_to(a.results))]=hashlib.sha256(f.read_bytes()).hexdigest()
out['scope_limitations']=['Audit v5 noise lacks confusion counts and HR windows; its stored metrics are aggregated only.','Clinical HR MAE is aggregated from stored record values; reference windows not retained in these files.','Memory and historical baseline measurements require separate source evidence.','No physical hardware validation or independent patient-level inference.']
a.out.write_text(json.dumps(out,indent=2,allow_nan=False));print('PASS:',len(r),'v6 trials;',count,'aggregate metrics checked')
