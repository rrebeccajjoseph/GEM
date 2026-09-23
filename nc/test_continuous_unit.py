"""CPU unit test of the cell-free objective: quadrature, continuous_nll, and
an end-to-end energy.train --continuous smoke run on synthetic data."""
import os, sys, math, json, tempfile, subprocess
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.getcwd())
from energy.quadrature import (fibonacci_sphere, random_rotation, unit_to_latlng, sample_vmf,
                               log_vmf, SphereQuadrature)
from energy.losses import continuous_nll
from energy.model import EnergyModel, latlng_to_unit_sphere
from energy.fields import RasterFields
from energy.grid import build_grid

torch.manual_seed(0)
rng = np.random.default_rng(0)

# --- quadrature -----------------------------------------------------------
v = fibonacci_sphere(5000)
assert torch.allclose(v.norm(dim=1), torch.ones(5000, dtype=torch.float64))
assert v.mean(0).abs().max() < 1e-3
R = random_rotation(rng)
assert torch.allclose(R @ R.T, torch.eye(3, dtype=torch.float64), atol=1e-12)
assert abs(torch.det(R).item() - 1) < 1e-12
ll = unit_to_latlng(v)
assert torch.allclose(latlng_to_unit_sphere(ll), v, atol=1e-9)

# unbiased: a von Mises-Fisher bump integrates to 4π sinh(κ)/κ; a small
# rotated node set gets it right on average even when a fixed one would not
kappa, mu = 200.0, torch.tensor([0.3, -0.5, 0.8124], dtype=torch.float64)
mu = mu / mu.norm()
true_log = math.log(4 * math.pi) + kappa - math.log(2 * kappa)  # sinh(κ) ≈ e^κ / 2
base = fibonacci_sphere(300)
est = []
for _ in range(4000):
    pts = base @ random_rotation(rng).T
    est.append(torch.logsumexp(kappa * (pts @ mu), 0).item() + math.log(4 * math.pi / 300))
mean_z = np.log(np.mean(np.exp(np.array(est) - true_log))) + true_log
assert abs(mean_z - true_log) < 0.05, (mean_z, true_log)

# vMF sampler: angular spread matches the kernel scale; log_vmf integrates to 1
mu1 = np.tile(np.array([[0.0, 0.0, 1.0]]), (20000, 1))
smp = sample_vmf(mu1, (6371.0 / 20.0) ** 2, rng)
assert np.allclose(np.linalg.norm(smp, axis=1), 1)
theta_km = np.arccos(np.clip(smp[:, 2], -1, 1)) * 6371.0
assert abs(np.sqrt((theta_km ** 2).mean() / 2) - 20.0) < 1.0     # per-axis std ≈ 20 km
dense_v = fibonacci_sphere(400000).float()
mass = torch.logsumexp(log_vmf(dense_v, torch.tensor([[0.0, 0.0, 1.0]]), 30.0)[:, 0], 0) \
    + math.log(4 * math.pi / 400000)
assert abs(mass.item()) < 1e-3

# the importance-sampled mixture is unbiased too, and on a bump far sharper
# than the uniform node spacing it is accurate where uniform nodes are hopeless
bump_ll = torch.tensor([[40.0, -3.7]])
bump_mu = latlng_to_unit_sphere(bump_ll).double()[0]
kappa = (6371.0 / 15.0) ** 2                                      # a 15 km bump
true_log = math.log(2 * math.pi / kappa)                          # ∫ exp(κ(mu·y - 1)) dy
q300 = SphereQuadrature(300)
def log_zhat(latlng, log_q):
    y = latlng_to_unit_sphere(latlng.double())
    return torch.logsumexp(kappa * (y @ bump_mu - 1) - log_q.double(), 0).item() - math.log(len(y))
mix = [log_zhat(*q300.sample(rng, bump_ll)) for _ in range(300)]
uni = [log_zhat(q300.rotated(rng), torch.full((300,), -math.log(4 * math.pi))) for _ in range(300)]
assert abs(np.median(mix) - true_log) < 0.2, (np.median(mix), true_log)
assert np.median(uni) < true_log - 5, 'uniform nodes should badly miss a sharp bump'
print('quadrature ok')

# --- continuous_nll ------------------------------------------------------
res = 1.0
H, W = 180, 360
lat_c = 90 - (np.arange(H) + 0.5) * res
fields_np = {
    'climate': np.tile((np.abs(lat_c) // 6)[:, None], (1, W)).astype(np.float32),
    'temp': np.tile((30 - 0.5 * np.abs(lat_c))[:, None], (1, W)).astype(np.float32),
}
fields_np['temp'][:5] = np.nan   # no data at the north pole cap
fields = RasterFields(fields_np, res)
cells, grid_latlngs = build_grid(1)
grid_t = torch.from_numpy(grid_latlngs).float()
table = {k: fields.sample(k, grid_t, 'nearest' if k == 'climate' else 'bilinear').numpy()
         .astype(np.float64) for k in fields_np}

model = EnergyModel(in_dim=16, d=32, n_masks=2, raster_table=table, use_season=True, gated=True)
f = torch.randn(8, 16)
y = torch.tensor(rng.uniform([-60, -180], [60, 180], (8, 2)), dtype=torch.float32)
nodes, log_q = SphereQuadrature(3000).sample(rng, y)
nll, neg_f_t, log_z, ess = continuous_nll(model, f, y, nodes, node_log_q=log_q,
                                          fields=fields, chunk_size=1000)
assert torch.isfinite(nll).all() and (ess >= 1).all()
nll.mean().backward()
for name in ('location_tower.net.0.weight', 'image_tower.net.0.weight', 'rasters.gate.weight'):
    assert dict(model.named_parameters())[name].grad.abs().sum() > 0, name
# chunking and checkpointing do not change the value
a = continuous_nll(model, f, y, nodes, node_log_q=log_q, fields=fields, chunk_size=5000)[0]
b = continuous_nll(model, f, y, nodes, node_log_q=log_q, fields=fields, chunk_size=700,
                   checkpoint_chunks=True)[0]
assert torch.allclose(a, b, atol=1e-4)
try:
    continuous_nll(model, f, y, nodes, fields=None)
    raise AssertionError('rasters without fields should raise')
except ValueError:
    pass
print('continuous_nll ok')

# --- no spike exploitation -----------------------------------------------
# Train on 4 tight clusters with a small node budget, then compare the
# training-time estimate of log Z with an accurate one (the same unbiased
# estimator with a far larger draw). Uniform nodes alone must show the
# failure — mass hidden in spikes between nodes — or the test is not
# sensitive; the importance-sampled proposal must not.
centers = torch.tensor([[48.8, 2.3], [35.7, 139.7], [-33.9, 151.2], [40.7, -74.0]])
emb = torch.randn(4, 16) * 3
big = SphereQuadrature(20000, local_per_target=400)

def train_and_gap(importance: bool):
    torch.manual_seed(1)
    r = np.random.default_rng(1)
    m = EnergyModel(in_dim=16, d=32)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    q = SphereQuadrature(2000)
    for step in range(300):
        c = torch.from_numpy(r.integers(0, 4, 64))
        yb = centers[c] + torch.randn(64, 2) * 0.3
        if importance:
            nodes, lq = q.sample(r, yb)
        else:
            nodes, lq = q.rotated(r), None
        nll, _, _, ess = continuous_nll(m, emb[c], yb, nodes, node_log_q=lq, chunk_size=4000)
        opt.zero_grad(); nll.mean().backward(); opt.step()
    with torch.no_grad():
        _, _, log_z_true, _ = continuous_nll(m, emb, centers, *big.sample(r, centers), chunk_size=8192)
        ests = []
        for _ in range(20):
            nodes, lq = q.sample(r, centers) if importance else (q.rotated(r), None)
            ests.append(continuous_nll(m, emb, centers, nodes, node_log_q=lq)[2])
        log_z_train = torch.stack(ests).median(0).values
    return (log_z_true - log_z_train).max().item(), nll.mean().item(), ess.median().item()

gap_u, nll_u, ess_u = train_and_gap(importance=False)
gap_i, nll_i, ess_i = train_and_gap(importance=True)
print(f'  uniform nodes only:  hidden log-mass {gap_u:8.2f} nats (train nll {nll_u:.1f}, ess {ess_u:.0f})')
print(f'  importance-sampled:  hidden log-mass {gap_i:8.2f} nats (train nll {nll_i:.1f}, ess {ess_i:.0f})')
assert gap_u > 5, 'test not sensitive: uniform-only training should hide mass in spikes'
assert gap_i < 0.5, 'importance-sampled training hid mass from its own Z estimate'
print('no spike exploitation ok')

# --- energy.train --continuous end to end --------------------------------
tmp = tempfile.mkdtemp()
np.savez_compressed(os.path.join(tmp, 'grid.npz'), cells=np.array([str(c) for c in cells]),
                    latlngs=grid_latlngs, resolution=1,
                    **{f'raster_{k}': v for k, v in table.items()})
np.savez_compressed(os.path.join(tmp, 'fields.npz'), res_deg=res,
                    **{f'field_{k}': v for k, v in fields_np.items()})
cache = os.path.join(tmp, 'cache'); os.makedirs(cache)
n = 600
c = rng.integers(0, 4, n)
lat_lng = centers.numpy()[c] + rng.normal(0, 0.5, (n, 2))
np.save(os.path.join(cache, 'embeddings.f16.npy'),
        (emb.numpy()[c] + rng.normal(0, 0.1, (n, 16))).astype(np.float16))
pd.DataFrame({'id': np.arange(n).astype(str), 'lat': lat_lng[:, 0], 'lng': lat_lng[:, 1],
              'selection': np.where(np.arange(n) < 500, 'train', 'val')}) \
    .to_csv(os.path.join(cache, 'index.csv'), index=False)
out = os.path.join(tmp, 'out')
base = [sys.executable, '-m', 'energy.train', '--cache', cache,
        '--grid', os.path.join(tmp, 'grid.npz'), '--fields', os.path.join(tmp, 'fields.npz'),
        '--out', out, '--run-name', 'smoke', '--continuous', '--rasters', '--gate', '--season',
        '--masks', '2', '--d', '32', '--quad-points', '3000', '--batch-size', '64',
        '--eval-samples', '100', '--chunk-size', '1500', '--warmup-steps', '5']
r = subprocess.run(base + ['--epochs', '3'], capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-3000:]
hist = json.load(open(os.path.join(out, 'smoke_history.json')))
print('  epochs:', [(h['epoch'], round(h['train_loss'], 2), round(h['median_km'], 1)) for h in hist])
assert hist[-1]['train_loss'] < hist[0]['train_loss']
assert 'visibility_temp' in hist[-1]
# resume from the last checkpoint runs further without error
r = subprocess.run(base + ['--epochs', '4', '--resume', os.path.join(out, 'smoke_last.pt')],
                   capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-3000:]
assert len(json.load(open(os.path.join(out, 'smoke_history.json')))) == 4
print('energy.train --continuous ok')
print('ALL OK')
