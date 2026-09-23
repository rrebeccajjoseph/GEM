"""Tiny CPU unit test of the rewritten Stage E pieces on a synthetic setup."""
import os, sys, tempfile, json, time
import numpy as np, torch
sys.path.insert(0, os.getcwd())
from energy.grid import build_grid, cell_index_map, _to_cell
from energy.refine import (CandidateBank, compute_targets, train_refiner,
                           refine_predictions, JointMLPScorer)
from energy.evaluation import distance_metrics

cells, latlngs = build_grid(4)
idx_map = cell_index_map(cells)
G = len(cells)
tmp = tempfile.mkdtemp()
t0 = time.time()
# Small table: coarse res 4 -> fine res 6 (49 children) keeps the build fast
bank = CandidateBank(cells, 4, 6, table_dir=tmp, n_workers=8)
print('table build', round(time.time() - t0, 1), 's', bank.ids.shape, bank.latlng.shape,
      'nan rows', int(np.isnan(bank.latlng[..., 0]).sum()))
# pentagon check: some rows must have fewer than 49 valid children
nvalid = (~np.isnan(bank.latlng[..., 0])).sum(1)
print('valid children min/max', nvalid.min(), nvalid.max(), 'n<49:', int((nvalid < 49).sum()))

rng = np.random.default_rng(0)
n, K = 300, 5
lat = rng.uniform(-60, 70, n).astype(np.float32); lng = rng.uniform(-180, 180, n).astype(np.float32)
ll = np.stack([lat, lng], 1)
true_parent = np.array([idx_map[_to_cell(float(a), float(b), 4)] for a, b in ll])
topk = rng.integers(0, G, (n, K))
# cover ~80% of rows by planting the true parent in a random slot
plant = rng.random(n) < 0.8
topk[plant, rng.integers(0, K, plant.sum())] = true_parent[plant]
targets = compute_targets(ll, topk, bank, idx_map)
print('covered frac', (targets >= 0).mean(), 'expected ~', plant.mean())
# verify targets: candidate latlng at target should be the true fine cell centroid
W = bank.n_children
ok = 0
for r in np.flatnonzero(targets >= 0)[:50]:
    slot, off = divmod(int(targets[r]), W)
    assert topk[r, slot] == true_parent[r]
    import h3
    assert h3.int_to_str(int(bank.ids[topk[r, slot], off])) == _to_cell(float(lat[r]), float(lng[r]), 6)
    ok += 1
print('target checks ok', ok)

emb = rng.standard_normal((n, 32)).astype(np.float16)
scorer = JointMLPScorer(in_dim=32, hidden=16)
logp = np.log(np.full((n, K), 1.0 / K, dtype=np.float32))
va = np.arange(40)
def eval_fn(s):
    out = refine_predictions(s, emb[va], topk[va], logp[va], bank, device='cpu')
    m = distance_metrics(out['pred_latlng'], ll[va]); m['mean_topk_coverage'] = float(out['coverage'].mean())
    return m
stats = train_refiner(scorer, emb, topk, targets, bank, topk_logp=logp, epochs=2, batch_size=16, cand_samples=64,
                      device='cpu', eval_fn=eval_fn, out_dir=tmp, run_name='t', ckpt_every=3,
                      stop_after_steps=7)
print('leg1', {k: v for k, v in stats.items() if k != 'history'})
ck = torch.load(os.path.join(tmp, 't_last.pt')); print('ckpt epoch/step', ck['epoch'], ck['step_in_epoch'], ck['global_step'])
scorer2 = JointMLPScorer(in_dim=32, hidden=16)
stats = train_refiner(scorer2, emb, topk, targets, bank, topk_logp=logp, epochs=2, batch_size=16, cand_samples=64,
                      device='cpu', eval_fn=eval_fn, out_dir=tmp, run_name='t', ckpt_every=3,
                      resume=os.path.join(tmp, 't_last.pt'))
print('leg2', {k: v for k, v in stats.items() if k != 'history'})
print('history epochs', [h['epoch'] for h in stats['history']])
assert os.path.exists(os.path.join(tmp, 't.pt')) and os.path.exists(os.path.join(tmp, 't_history.json'))
out = refine_predictions(scorer2, emb, topk, logp, bank, device='cpu', rows_per_batch=7)
assert np.isfinite(out['pred_latlng']).all() and np.isfinite(out['fine_entropy']).all()
print('entropy range', out['fine_entropy'].min(), out['fine_entropy'].max(), 'coverage', out['coverage'][:3])
print('UNIT OK')
