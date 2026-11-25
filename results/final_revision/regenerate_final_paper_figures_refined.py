"""Generate audit-based figures without modifying historical results. Run from repo root."""
import json, argparse, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser(); p.add_argument('--results',type=Path,default=Path('results')); p.add_argument('--out',type=Path,default=Path('results/final_paper_figures_refined')); a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
def load(v,n): return json.loads((a.results/f'audit_v{v}'/(n+'.json')).read_text())
r=load(6,'trials'); assert len(r)==530 and len({x['id'] for x in r})==530
assert all(sum(x[k] for k in ('tp','fp','fn','tn'))==6000 for x in r)
def group(g,s): return [x for x in r if x['group']==g and x['scenario']==s]
def stats(rows,k):
 v=[x[k] for x in rows if x.get(k) is not None]; return (st.mean(v),st.stdev(v) if len(v)>1 else 0) if v else (float('nan'),0)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'figure.dpi':150})
def save(fig,name):
 
 if len(fig.axes)>1:
  for i,ax in enumerate(fig.axes): ax.set_title(f'({chr(97+i)})',loc='left',fontweight='bold')
 fig.text(.5,.005,'Mean ± sample SD; dots, where shown: individual records. SD endpoints are not physical bounds.',ha='center',fontsize=7); fig.tight_layout(rect=(0,.05,1,1)); fig.savefig(a.out/(name+'.pdf'),bbox_inches='tight'); fig.savefig(a.out/(name+'.png'),dpi=600,bbox_inches='tight'); plt.close(fig)
def line(ax,x,groups,k,label):
 m,s=zip(*(stats(g,k) for g in groups)); ax.errorbar(x,m,yerr=s,marker='o',capsize=3,label=label)
def bars(ax,labels,groups,metrics):
 x=np.arange(len(labels)); w=.8/len(metrics)
 for i,(k,label) in enumerate(metrics):
  m,s=zip(*(stats(g,k) for g in groups)); pos=x+(i-(len(metrics)-1)/2)*w; ax.bar(pos,m,w,yerr=s,capsize=2,label=label)
  for j,v in enumerate(m):
   if np.isnan(v): ax.text(pos[j],0,'N/A',rotation=90,ha='center',va='bottom',fontsize=7)
  if any('record' in r for g in groups for r in g):
   for j,g in enumerate(groups):
    vals=[r[k] for r in g if r.get(k) is not None]; ax.scatter(pos[j]+np.linspace(-w*.24,w*.24,len(vals)),vals,s=9,c='black',alpha=.55,zorder=4)
 ax.set_xticks(x,labels); ax.legend(frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.18),ncol=min(4,len(metrics)))
def panel(name,labels,groups,metrics,ylabel):
 f,ax=plt.subplots(figsize=(7,3.2)); bars(ax,labels,groups,metrics); ax.set_ylabel(ylabel); save(f,name)
noise=[load(5,f'refractory_{n}_noise') for n in (30,48)]; sig=[0,.08,.15,.25,.35]
f,axs=plt.subplots(1,3,figsize=(9,3))
for rows,label in zip(noise,['300 ms','480 ms']):
 for ax,k,y in zip(axs,['peaks','fpr','rr_mae'],['Detected peaks','Fault FPR (%)','RR MAE (BPM)']):
  line(ax,sig,[[r for r in rows if r['sigma']==s] for s in sig],k,label); ax.set(xlabel='Noise SD (mV)',ylabel=y)
axs[0].legend(frameon=False); save(f,'noise_refractory')
g=[group('nstdb',s) for s in ['clean','bw','ma','em25','em35']]; labels=['Clean','BW\n0.25 mV','MA\n0.20 mV','EM\n0.25 mV','EM\n0.35 mV']
f,axs=plt.subplots(1,2,figsize=(9,3.4)); bars(axs[0],labels,g,[('rr_mae','RR MAE')]); axs[0].set_ylabel('BPM'); bars(axs[1],labels,g,[('f1','Fault F1'),('fpr','Fault FPR')]); axs[1].set_ylabel('%'); save(f,'nstdb')
for cat,keys,labels in [('single',['missing','burst','range','stale','drift'],['Missing','Burst','Range','Stale','Drift']),('compound',['nm','bs','nd','tc'],['Noise + missing','Burst + stale','Noise + drift','Triple'])]:
 panel(cat,labels,[group(cat,k) for k in keys],[('precision','Precision'),('recall','Recall'),('f1','Fault F1'),('fpr','FPR')],'%')
f,ax=plt.subplots(figsize=(5,3)); line(ax,list(range(10,15)),[group('boundary',str(k)) for k in range(10,15)],'boundary_flag_pct','Inconsistency flags'); ax.set(xlabel='Injected offset (BPM)',ylabel='Flagged injection samples (%)'); save(f,'boundary')
f,axs=plt.subplots(1,2,figsize=(8,3)); ds=[.01,.025,.05,.1,.2]
for d in ds:
 for metric in ['f1','fpr']:
  assert [r[metric] for r in group('density',f'missing_{d}')]==[r[metric] for r in group('density',f'range_{d}')]
for k in ['missing','drift']:
 g=[group('density',f'{k}_{d}') for d in ds]
 for ax,m in zip(axs,['f1','fpr']):
  if m=='fpr' and k=='drift':
   assert all([r[m] for r in group('density',f'missing_{d}')]==[r[m] for r in group('density',f'drift_{d}')] for d in ds); continue
  line(ax,[100*d for d in ds],g,m,'All three fault types' if m=='fpr' else ('Missing / range' if k=='missing' else 'Drift')); ax.set_xlabel('Fault density (%)')
axs[0].set_ylabel('Fault F1 (%)'); axs[1].set_ylabel('Fault FPR (%)'); axs[1].legend(frameon=False); axs[0].legend(frameon=False); save(f,'density')
c=load(5,'clinical_uncertainty'); keys=list(c); labels=[k.replace('_',' ').title() for k in keys]; g=[c[k]['records'] for k in keys]
panel('clinical_hr',labels,g,[('raw_mae','Raw'),('rr_mae','RR filtered'),('hold_mae','Hold last')],'MAE (BPM)')
panel('clinical_fault',labels,g,[('precision','Precision'),('recall','Recall'),('f1','Fault F1'),('fpr','FPR')],'%')
pt=load(5,'pt_style_clinical'); f,ax=plt.subplots(figsize=(7,3)); x=np.arange(len(keys))
for i,(data,label) in enumerate([(c,'Proposed'),(pt,'PT-style C')]):
 m,s=zip(*(stats(data[k]['records'],'rr_mae') for k in keys)); pos=x+(i-.5)*.35; ax.bar(pos,m,.35,yerr=s,capsize=3,label=label)
 for j,k in enumerate(keys):
  vals=[r['rr_mae'] for r in data[k]['records']]; ax.scatter(pos[j]+np.linspace(-.08,.08,len(vals)),vals,s=10,c='black',alpha=.55,zorder=4)
ax.set_xticks(x,labels); ax.set_ylabel('RR MAE (BPM)'); ax.legend(frameon=False); save(f,'clinical_pt_comparison')
print('Generated figures in',a.out)
