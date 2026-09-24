"""CPU unit test of the spatially held-out val split (val_split.py): no val
row has a train row within buffer_km, test rows are untouched, the per-block
cap spreads val over many blocks, and the split is deterministic."""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.getcwd())
from val_split import spatial_val_split, _unit, EARTH_RADIUS_KM
from scipy.spatial import cKDTree

# street-view-like sequences: 3000 walks of 40 frames ~30 m apart, with
# start points dense in a few "cities" and sparse elsewhere
rng = np.random.default_rng(0)
cities = rng.uniform([-50, -170], [60, 170], size=(20, 2))
starts = np.concatenate([cities[rng.integers(0, 20, 2000)] + rng.normal(0, 0.2, (2000, 2)),
                         rng.uniform([-50, -170], [60, 170], size=(1000, 2))])
step = rng.normal(0, 0.0003, (3000, 40, 2)).cumsum(axis=1)
pts = (starts[:, None, :] + step).reshape(-1, 2)
df = pd.DataFrame({'lat': pts[:, 0], 'lng': pts[:, 1], 'selection': 'train'})
df.loc[df.sample(5000, random_state=1).index, 'selection'] = 'test'

sel = spatial_val_split(df, val_size=2000, seed=330)
counts = sel.value_counts()
print(dict(counts))
assert counts['val'] == 2000
assert (sel[df['selection'] == 'test'] == 'test').all()            # test untouched
assert set(sel[df['selection'] == 'train']) <= {'train', 'val', 'val_buffer'}

# the guarantee: nearest remaining train row to any val row is > 1 km
xyz = _unit(df['lat'].to_numpy(), df['lng'].to_numpy())
d, _ = cKDTree(xyz[(sel == 'train').to_numpy()]).query(xyz[(sel == 'val').to_numpy()])
km = 2 * np.arcsin(d / 2) * EARTH_RADIUS_KM
print(f'nearest train to val: min {km.min():.3f} km, median {np.median(km):.2f} km')
assert km.min() > 1.0

# a random split of the same data would leak: the check above has teeth
rand = df[df['selection'] == 'train'].sample(2000, random_state=2).index
rest = df.index.difference(rand).difference(df.index[df['selection'] == 'test'])
d, _ = cKDTree(xyz[rest]).query(xyz[rand])
assert np.median(2 * np.arcsin(d / 2) * EARTH_RADIUS_KM) < 0.1

# per-block cap spreads val out; buffer costs a bounded slice of train
blocks = set(zip(np.floor(df.loc[sel == 'val', 'lat'] / 0.1), np.floor(df.loc[sel == 'val', 'lng'] / 0.1)))
assert len(blocks) >= 2000 // 20
assert counts['val_buffer'] < 0.2 * (df['selection'] == 'train').sum()

assert sel.equals(spatial_val_split(df, val_size=2000, seed=330))  # deterministic
print('val split ok')
