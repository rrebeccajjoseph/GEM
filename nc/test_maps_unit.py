"""CPU unit test of energy.maps on synthetic GeoJSON: rasterization, drive
side, geodesic coast distance, and the grid/fields it writes loading into a
model — including a country too small to win any grid cell."""
import os, sys, json, tempfile, subprocess, math
import numpy as np, torch
sys.path.insert(0, os.getcwd())
from energy.maps import (rasterize_categorical, remap_dense, coast_distance_km,
                         NE_FILES, LABEL_DERIVED)
from energy.fields import RasterFields
from energy.train import load_grid
from energy.model import EnergyModel
from energy.grid import build_grid


def box(lng0, lat0, lng1, lat1):
    return {'type': 'Polygon', 'coordinates': [[[lng0, lat0], [lng1, lat0], [lng1, lat1],
                                                [lng0, lat1], [lng0, lat0]]]}


def feature(geom, **props):
    return {'type': 'Feature', 'geometry': geom, 'properties': props}


# two big countries side by side, one tiny island nation, all on one continent
admin0 = [feature(box(0.07, -10, 20, 10), ADM0_A3='GBR'),   # left-hand traffic
          feature(box(20, -10, 40, 10), ADM0_A3='FRA'),
          feature(box(60.0, 30.0, 60.3, 30.3), ADM0_A3='MLT')]  # smaller than a res-1 cell
admin1 = [feature(box(0, -10, 20, 0), adm1_code='GBR-S'), feature(box(0, 0, 20, 10), adm1_code='GBR-N'),
          feature(box(20, -10, 40, 10), adm1_code='FRA-1'),
          feature(box(60.0, 30.0, 60.3, 30.3), adm1_code='MLT-1')]
land = [feature(box(0, -10, 40, 10), featurecla='Land'), feature(box(60.0, 30.0, 60.3, 30.3))]

res = 0.1
field, labels = rasterize_categorical(admin0, lambda f: f['properties']['ADM0_A3'], res)
rf = RasterFields({'c': field}, res)
pts = torch.tensor([[0.0, 10.0], [0.0, 30.0], [30.15, 60.15], [50.0, 50.0]])
got = [None if math.isnan(v) else labels[int(v)] for v in rf.sample('c', pts, 'nearest').tolist()]
assert got == ['GBR', 'FRA', 'MLT', None], got

# all_touched: a pixel whose centre is just outside the polygon still gets the label
edge = torch.tensor([[10.0, 0.03]])       # pixel [0.0, 0.1) is centred at 0.05, box starts at 0.07
assert labels[int(rf.sample('c', edge, 'nearest').item())] == 'GBR'
strict, _ = rasterize_categorical(admin0, lambda f: f['properties']['ADM0_A3'], res, all_touched=False)
assert math.isnan(RasterFields({'c': strict}, res).sample('c', edge, 'nearest').item())

d, codes = remap_dense(np.array([[10, 220, np.nan], [10, 50, 220]], dtype=np.float32))
assert codes == [10, 50, 220] and np.array_equal(d[~np.isnan(d)], [0, 2, 0, 1, 2])
print('rasterize ok')

# geodesic coast distance: the box centre is 10 deg (~1112 km) from its
# nearest (north/south) edge; an equirectangular transform would not care
# about latitude, this does
land_mask = ~np.isnan(rasterize_categorical(land, lambda f: 'land', res, all_touched=False)[0])
coast = coast_distance_km(land_mask, res)
cr = RasterFields({'k': coast}, res)
at_centre = cr.sample('k', torch.tensor([[0.0, 20.0]]), 'bilinear').item()
assert abs(at_centre - 10 * 111.19) < 20, at_centre
assert math.isnan(cr.sample('k', torch.tensor([[50.0, 50.0]]), 'nearest').item())  # sea
print('coast distance ok')

# --- energy.maps end to end -----------------------------------------------
tmp = tempfile.mkdtemp()
ne = os.path.join(tmp, 'ne'); os.makedirs(ne)
for key, feats in (('admin0', admin0), ('admin1', admin1), ('land', land)):
    json.dump({'type': 'FeatureCollection', 'features': feats}, open(os.path.join(ne, NE_FILES[key]), 'w'))
cells, ll = build_grid(1)
G = len(cells)
rng = np.random.default_rng(0)
np.savez_compressed(os.path.join(tmp, 'grid.npz'), cells=np.array([str(c) for c in cells]),
                    latlngs=ll, resolution=1,
                    raster_climate=rng.integers(0, 30, G).astype(float),
                    raster_scene=rng.integers(0, 365, G).astype(float),
                    raster_country=rng.integers(0, 50, G).astype(float))
climate = np.tile((np.arange(1800) // 60 % 30)[:, None], (1, 3600)).astype(np.float32)
np.savez_compressed(os.path.join(tmp, 'fields.npz'), res_deg=0.1, field_climate=climate)
r = subprocess.run([sys.executable, '-m', 'energy.maps', '--ne-dir', ne,
                    '--grid', os.path.join(tmp, 'grid.npz'), '--out-grid', os.path.join(tmp, 'maps.npz'),
                    '--fields', os.path.join(tmp, 'fields.npz')], capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-3000:]

_, _, table = load_grid(os.path.join(tmp, 'maps.npz'), want_rasters=True)
assert set(table) == {'climate', 'country', 'region', 'drive_side', 'coast_km'}, sorted(table)
assert not (set(table) - {'country', 'region'}) & LABEL_DERIVED
assert table['country'][1] == 3 and table['region'][1] == 4      # full label counts
# MLT is smaller than every res-1 cell: it wins no cell in the table...
assert np.nanmax(table['country'][0]) < 2
fields = RasterFields.load(os.path.join(tmp, 'fields.npz'))
assert set(fields.names) == {'climate', 'country', 'region', 'drive_side', 'coast_km'}

m = EnergyModel(in_dim=16, d=32, raster_table=table, use_season=True, gated=True)
assert m.rasters.n_classes['country'] == 3 and m.rasters.n_classes['region'] == 4
# ...but sampled at its own coordinates it is still a valid class
vals = m.rasters.at_points(fields, torch.tensor([[30.15, 60.15], [0.0, 10.0], [0.0, 30.0]]))
assert vals['country'][1].all() and vals['drive_side'][1].all()
assert vals['drive_side'][0].tolist() == [1.0, 1.0, 0.0]            # MLT, GBR left; FRA right
f = torch.randn(3, 16)
out = m.neg_free_energy(f, m.location_tower(torch.tensor([[30.15, 60.15], [0.0, 10.0], [0.0, 30.0]])),
                        raster_values=vals)
assert torch.isfinite(out).all()
print('energy.maps end to end ok')
print('ALL OK')
