"""Full-size parity of the native ArchesWeatherGen vs the official geoarches DiffusionModule (oracle). Needs the geoarches modelstore layout
(configs from geoarches/paper/configs + checkpoints symlinked under modelstore/<name>/checkpoints/last.ckpt). Run from that directory."""
import sys, time, torch, warnings, numpy as np
warnings.filterwarnings('ignore')
from tensordict.tensordict import TensorDict
from geoarches.lightning_modules.base_module import load_module
from weatherai.models.arches import load_official_gen, legacy_overflow_time_features
torch.manual_seed(0)
og, _ = load_module('archesweathergen')
ng, st = load_official_gen('/workspace/ckpt/arches')
def rs(): return {'surface': torch.randn(1,4,1,121,240)*0.7, 'level': torch.randn(1,6,13,121,240)*0.7}
s, p = rs(), rs()
td = lambda d: TensorDict(surface=d['surface'], level=d['level'], batch_size=1)
month, hour = torch.tensor([6]), torch.tensor([12])
ts = torch.tensor([1591012800])  # 2020-06-01 12Z
batch = dict(state=td(s), prev_state=td(p), month=month, hour_of_day=hour, timestamp=ts, lead_time_hours=torch.tensor([24]))
def md(a,b):
    return max((a[k]-b[k]).abs().max().item() for k in ('surface','level')), max(((a[k]-b[k]).norm()/b[k].norm()).item() for k in ('surface','level'))
with torch.no_grad():
    t=time.time(); od = og.det_model.core[0](batch); print('oracle det', time.time()-t)
    t=time.time(); nd = ng.det_model.core[0](s, p, month, hour); print('native det', time.time()-t)
    print('DET member0  max|d|, rel:', md(nd, od))
    od_avg = og.det_model(batch); nd_avg = ng.det_model(s,p,month,hour)
    print('DET avg of 4 max|d|, rel:', md(nd_avg, od_avg))
    # month/hour legacy features vs pandas path
    import pandas as pd
    for tsv in [1591012800, 1700000000, 1234567890, 1000000000, 1900000000]:
        t32 = torch.tensor([tsv]).cpu().to(torch.int32).numpy() * 10**9
        tt = pd.to_datetime(t32).tz_localize(None); m,h = legacy_overflow_time_features([tsv])
        print('legacy feats', tsv, (tt.month.tolist(), tt.hour.tolist()), (m.tolist(), h.tolist()))
    # gen network, one call (is_sampling)
    noisy = rs(); tstep = torch.tensor([700.0])
    o = og.forward(batch, td(noisy), tstep, pred_state=od_avg, is_sampling=True)
    ng._timestamp = ts
    n = ng.predict(s, noisy, tstep, nd_avg, p, month, hour, is_sampling=True); ng._timestamp=None
    print('GEN net (is_sampling) max|d|, rel:', md(n, o))
    # full sampling with the same noise
    g = torch.Generator().manual_seed(5)
    noise = {k: torch.empty_like(s[k]).normal_(generator=g) for k in ('surface','level')}
    og.device  # ensure attr
    t=time.time(); of = og.sample(batch, seed=5, num_steps=5, disable_tqdm=True); print('oracle sample', time.time()-t)
    t=time.time(); nf = ng.sample(s, p, month, hour, timestamp=ts, num_steps=5, seed=5); print('native sample', time.time()-t)
    print('SAMPLE (5 steps, seed 5) max|d|, rel:', md(nf, of))
    print('ref std', of['surface'].std().item(), of['level'].std().item())
