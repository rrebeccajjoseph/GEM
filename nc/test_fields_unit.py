"""CPU unit test of lat/lng raster fields and RasterBank's point path."""
import os, sys, tempfile
import numpy as np, torch
sys.path.insert(0, os.getcwd())
from energy.fields import RasterFields
from energy.model import RasterBank, EnergyModel
from energy.grid import build_grid, build_field, RasterioSampler

torch.manual_seed(0)
rng = np.random.default_rng(0)

# --- RasterFields sampling on a 1-degree field ---------------------------
res = 1.0
H, W = 180, 360
arr = rng.normal(size=(H, W)).astype(np.float32)
rf = RasterFields({'a': arr}, res)

i, j = rng.integers(0, H, 500), rng.integers(0, W, 500)
centers = torch.tensor(np.stack([90 - (i + 0.5) * res, -180 + (j + 0.5) * res], 1),
                       dtype=torch.float32)
for mode in ('nearest', 'bilinear'):
    got = rf.sample('a', centers, mode).numpy()
    assert np.allclose(got, arr[i, j], atol=1e-5), f'{mode} not exact at pixel centers'

# bilinear midway between two horizontal neighbours = their mean
mid = torch.tensor([[89.5 - 10, -180 + 20.5 + 0.5]])   # between cols 20 and 21, row 10
assert np.isclose(rf.sample('a', mid).item(), arr[10, 20:22].mean(), atol=1e-5)

# longitude wraps: 190 == -170, and the antimeridian blends col 359 with col 0
assert np.allclose(rf.sample('a', torch.tensor([[0.5, 190.5]])).numpy(),
                   rf.sample('a', torch.tensor([[0.5, -169.5]])).numpy())
assert np.isclose(rf.sample('a', torch.tensor([[89.5, 180.0]])).item(),
                  (arr[0, 359] + arr[0, 0]) / 2, atol=1e-5)

# latitude clamps at the poles instead of indexing out of range
assert torch.isfinite(rf.sample('a', torch.tensor([[95.0, 0.0], [-95.0, 0.0]]))).all()

# NaN corners are renormalized away; all-NaN corners give NaN
arr2 = arr.copy()
arr2[10, 21] = np.nan
rf2 = RasterFields({'a': arr2}, res)
assert np.isclose(rf2.sample('a', mid).item(), arr2[10, 20], atol=1e-5)
arr2[10, 20] = np.nan
rf2 = RasterFields({'a': arr2}, res)
assert np.isnan(rf2.sample('a', mid).item())
assert np.isnan(rf2.sample('a', torch.tensor([[89.5 - 10, -180 + 21.5]]), 'nearest').item())
print('RasterFields sampling ok')

# --- RasterBank: table path unchanged, point path consistent ------------
G = 2000
table = {
    'climate': np.where(rng.random(G) < 0.3, np.nan, rng.integers(0, 30, G)).astype(np.float64),
    'popdens': np.where(rng.random(G) < 0.3, np.nan, rng.exponential(200, G)),
    'temp': np.where(rng.random(G) < 0.3, np.nan, rng.normal(15, 10, G)),
    'drive_side': np.where(rng.random(G) < 0.3, np.nan, rng.integers(0, 2, G)).astype(np.float64),
    'country': rng.integers(0, 50, G).astype(np.float64),
}
bank = RasterBank(table, in_dim=16)

# buffers equal the pre-refactor normalization (reimplemented here)
for name, raw in table.items():
    raw = raw.copy()
    valid = ~np.isnan(raw)
    if name in ('climate', 'country'):
        ref = np.where(valid, raw, 0).astype(np.int64)
    elif name == 'drive_side':
        ref = np.where(valid, raw, 0.0).astype(np.float32)
    else:
        if name == 'popdens':
            raw[valid] = np.log1p(np.maximum(raw[valid], 0.0))
        mean, std = raw[valid].mean(), raw[valid].std() + 1e-8
        ref = np.where(valid, (raw - mean) / std, 0.0).astype(np.float32)
    assert np.array_equal(getattr(bank, f'values_{name}').numpy(), ref), name
    assert np.array_equal(getattr(bank, f'valid_{name}').numpy(), valid), name

f = torch.randn(8, 16)
s = slice(100, 900)
assert torch.equal(bank(f, s), bank(f, values=bank.lookup(s)))

# out-of-range class codes from a field are masked, not indexed
vals, valid = bank.normalize('climate', torch.tensor([3.0, 31.0, -1.0, float('nan')]))
assert valid.tolist() == [True, False, False, False] and vals.max() < 30
print('RasterBank table/point paths ok')

# --- EnergyModel: raster_values over points == grid path at those rows --
# Fields whose pixel at each chosen point holds exactly that grid row's
# table value: sampling the points must then reproduce the grid path.
res = 1.0
pts_i, pts_j = rng.choice(H, 64, replace=False), rng.choice(W, 64, replace=False)
rows = np.arange(64)
latlng = torch.tensor(np.stack([90 - (pts_i + 0.5) * res, -180 + (pts_j + 0.5) * res], 1),
                      dtype=torch.float32)
fields = {}
for name, raw in table.items():
    a = np.full((H, W), np.nan, dtype=np.float32)
    a[pts_i, pts_j] = raw[rows]
    fields[name] = a
# isolated pixels: bilinear at their exact centers returns them unchanged
rf = RasterFields(fields, res)
model = EnergyModel(in_dim=16, d=32, n_masks=4, raster_table=table, use_season=True).eval()
loc = model.location_tower(latlng)
with torch.no_grad():
    grid_path = model.neg_free_energy(f, loc, grid_slice=slice(0, 64))
    point_path = model.neg_free_energy(f, loc, raster_values=model.rasters.at_points(rf, latlng))
assert torch.allclose(grid_path, point_path, atol=1e-4), (grid_path - point_path).abs().max()

try:
    model.rasters.at_points(RasterFields({'temp': fields['temp']}, res), latlng)
    raise AssertionError('missing fields should raise')
except KeyError:
    pass
print('EnergyModel point path ok')

# --- build_field on a synthetic GeoTIFF ---------------------------------
import rasterio
from rasterio.transform import from_origin

tmp = tempfile.mkdtemp()
src = rng.normal(size=(360, 720)).astype(np.float32)   # 0.5 deg
src[:20, :] = -9999.0                                  # nodata band at the top
cls = rng.integers(1, 4, (360, 720)).astype(np.float32)
cls[::2, ::2] = cls[1::2, 1::2] = cls[::2, 1::2] = 2  # 3 of every 2x2 block
for name, data, nodata in (('cont.tif', src, -9999.0), ('cat.tif', cls, 0.0)):
    with rasterio.open(os.path.join(tmp, name), 'w', driver='GTiff', height=360, width=720,
                       count=1, dtype='float32', crs='EPSG:4326', nodata=nodata,
                       transform=from_origin(-180, 90, 0.5, 0.5)) as ds:
        ds.write(data, 1)

cont = build_field(RasterioSampler(os.path.join(tmp, 'cont.tif')), 'mean', 1.0)
blocks = src.reshape(180, 2, 360, 2).mean(axis=(1, 3))
assert cont.shape == (180, 360)
assert np.isnan(cont[:10]).all(), 'nodata rows should stay NaN'
assert np.allclose(cont[10:], blocks[10:], atol=1e-4), 'mean resampling != 2x2 block mean'

cat = build_field(RasterioSampler(os.path.join(tmp, 'cat.tif'), nodata=0,
                                  transform_value=lambda v: v - 1), 'modal', 1.0)
assert (cat == 1).all(), 'mode resampling + transform_value should give 2 - 1 everywhere'
print('build_field ok')

# --- visibility gate ------------------------------------------------------
gated = RasterBank(table, in_dim=16, gated=True)
gated.load_state_dict(bank.state_dict(), strict=False)   # same heads, fresh gate
s = slice(0, 500)
plain, g = bank(f, s), gated(f, s)
# a gate floors each term at log(1 - pi): at init (logit 10) the gated bank
# equals the ungated one wherever the summed logit is well above that floor
far = plain > -5
assert torch.allclose(g[far], plain[far], atol=1e-2), (g - plain)[far].abs().max()  # per-term floors, summed
assert (g >= plain - 1e-4).all()          # the gate only ever lifts a term
# pi -> 0: every term is flat in y (all zeros)
torch.nn.init.constant_(gated.gate.bias, -30.0)
assert gated(f, s).abs().max() < 1e-6
# no-data points stay exactly 0 for any pi
torch.nn.init.normal_(gated.gate.weight)
only_temp = RasterBank({'temp': table['temp']}, in_dim=16, gated=True)
torch.nn.init.normal_(only_temp.gate.weight)
out = only_temp(f)
assert (out[:, ~only_temp.valid_temp] == 0).all()
# gate gradients flow; ungated checkpoints still load into ungated models
out.sum().backward()
assert only_temp.gate.weight.grad is not None and only_temp.gate.weight.grad.abs().sum() > 0
print('visibility gate ok')
print('ALL OK')
