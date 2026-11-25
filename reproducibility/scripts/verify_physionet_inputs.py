"""Run from repository root in the existing WFDB virtual environment. Read-only cache comparison."""
import hashlib, tempfile, pathlib, urllib.request, json
import numpy as np
import wfdb
sets={'mitdb':'100 101 102 103 105 106 108 113 115 119 200'.split(),'nstdb':['bw','ma','em']}
rows=[]
with tempfile.TemporaryDirectory() as tmp:
 for db,recs in sets.items():
  root=pathlib.Path(tmp)/db;root.mkdir()
  base=f'https://physionet.org/files/{db}/1.0.0/'
  sums={line.split()[1].lstrip('*'):line.split()[0] for line in urllib.request.urlopen(base+'SHA256SUMS').read().decode().splitlines() if len(line.split())>=2}
  for rec in recs:
   for ext in (['hea','dat','atr'] if db=='mitdb' else ['hea','dat']):
    name=f'{rec}.{ext}';path=root/name;urllib.request.urlretrieve(base+name,path)
    actual=hashlib.sha256(path.read_bytes()).hexdigest();expected=sums.get(name);assert actual==expected,(name,actual,expected)
   x=wfdb.rdrecord(str(root/rec),sampto=21600).p_signal[:,0]
   cache=pathlib.Path('data_cache')/f'{db}_{rec}_dat.npz'
   ok=np.array_equal(np.load(cache)['sig'].reshape(-1),x) if cache.exists() else None
   ann=None
   if db=='mitdb':
    a=wfdb.rdann(str(root/rec),'atr',sampto=21600);c=pathlib.Path('data_cache')/f'{db}_{rec}_atr.npz'
    if c.exists():
     z=np.load(c,allow_pickle=True);mask=z['sample']<21600;ann=bool(np.array_equal(z['sample'][mask],a.sample) and np.array_equal(z['symbol'][mask],a.symbol))
   rows.append(dict(database=db,record=rec,source_checksums=True,signal_cache_equal=bool(ok) if ok is not None else None,annotation_cache_equal=ann))
   print(rows[-1],flush=True)
pathlib.Path('physionet_input_verification.json').write_text(json.dumps(rows,indent=2))
