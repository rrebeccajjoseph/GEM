"""CPU unit test of cell-free inference: NMS starts, raster-field gradients,
and gradient descent on the energy beating the best quadrature node."""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.getcwd())
from energy.model import EnergyModel
from energy.fields import RasterFields
from energy.quadrature import SphereQuadrature
from energy.losses import continuous_nll, haversine_km_torch
from energy.infer import nms_starts, predict_continuous, evaluate_continuous

torch.manual_seed(0)
rng = np.random.default_rng(0)

# --- NMS ------------------------------------------------------------------
nodes = SphereQuadrature(5000).fixed()
log_p = torch.log_softmax(torch.randn(3, 5000) * 3, dim=1)
starts = nms_starts(log_p, nodes, k=6, min_sep_km=500.0, pool=200)
assert starts.shape == (3, 6, 2)
for b in range(3):
    d = haversine_km_torch(starts[b], starts[b])
    uniq = torch.unique(starts[b], dim=0)
    dd = haversine_km_torch(uniq, uniq) + torch.eye(len(uniq)) * 1e9
    assert dd.min() >= 500.0
    assert torch.equal(starts[b, 0], nodes[log_p[b].argmax()])   # first start = argmax
print('nms ok')

# --- bilinear fields are differentiable in lat/lng ------------------------
arr = np.add.outer(np.linspace(0, 5, 180), np.sin(np.linspace(0, 6, 360))).astype(np.float32)
rf = RasterFields({'t': arr}, 1.0)
p = torch.tensor([[12.3, 45.6]], requires_grad=True)
rf.sample('t', p).sum().backward()
eps = 1e-3
for i in range(2):
    dp = torch.zeros(1, 2); dp[0, i] = eps
    fd = (rf.sample('t', p.detach() + dp) - rf.sample('t', p.detach() - dp)).item() / (2 * eps)
    assert abs(p.grad[0, i].item() - fd) < 1e-3, (i, p.grad, fd)
# near a no-data pixel the value is finite and so is its gradient
arr_nan = np.ones((180, 360), np.float32); arr_nan[:5] = np.nan
pn = torch.tensor([[85.2, 10.0]], requires_grad=True)
vn = RasterFields({'t': arr_nan}, 1.0).sample('t', pn)
vn.sum().backward()
assert torch.isfinite(vn).all() and torch.isfinite(pn.grad).all(), pn.grad
print('field gradients ok')

# --- refinement beats the node argmax on a trained model -----------------
# Clusters much tighter than the coarse node spacing (~500 km at 2000 nodes):
# the best node is off by hundreds of km, descent should land near the mode.
centers = torch.tensor([[48.8, 2.3], [35.7, 139.7], [-33.9, 151.2], [40.7, -74.0],
                        [-23.5, -46.6], [19.4, -99.1]])
C = len(centers)
emb = torch.randn(C, 16) * 3
m = EnergyModel(in_dim=16, d=64)
opt = torch.optim.Adam(m.parameters(), lr=2e-3)
q = SphereQuadrature(4000)
for step in range(600):
    c = torch.from_numpy(rng.integers(0, C, 128))
    yb = centers[c] + torch.randn(128, 2) * 0.1
    nodes_t, lq = q.sample(rng, yb)
    nll = continuous_nll(m, emb[c], yb, nodes_t, node_log_q=lq, chunk_size=8000)[0]
    opt.zero_grad(); nll.mean().backward(); opt.step()

coarse = SphereQuadrature(2000).fixed()
coarse_rff = m.location_tower.encode_features(coarse)
out = predict_continuous(m, emb, coarse_rff, coarse, k=4, steps=150)
assert (out['end_neg_f'] >= out['start_neg_f'] - 1e-5).all()      # never worse
node_err = haversine_km_torch(out['node_pred'], centers).diagonal()
ref_err = haversine_km_torch(out['pred'], centers).diagonal()
print('  node argmax error km:', [round(x) for x in node_err.tolist()])
print('  search+descent km:    ', [round(x) for x in ref_err.tolist()])
assert (ref_err < node_err).all(), 'refinement lost to the best node somewhere'
assert ref_err.max() < 50, 'refinement stopped short of a mode'

# descent alone (no local search) is the weaker baseline this guards against
plain = predict_continuous(m, emb, coarse_rff, coarse, k=4, steps=150, search_samples=0)
plain_err = haversine_km_torch(plain['pred'], centers).diagonal()
print('  descent only error km:', [round(x) for x in plain_err.tolist()])

metrics = evaluate_continuous(m, emb, centers.numpy(), coarse_rff, coarse, k=4, steps=150,
                              batch_size=4)
assert metrics['refined_median_km'] < metrics['coarse_median_km']
assert metrics['refined_mean_energy_drop'] >= 0
print('refinement ok')
print('ALL OK')
