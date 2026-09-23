"""CPU unit test of the train/benchmark leakage check: Flickr id and owner
parsing, the great-circle geo gate, and near-duplicate thresholding."""
import os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
from energy.overlap import flickr_photo_id, flickr_owner, geo_candidates, near_duplicates

# --- filename parsing -----------------------------------------------------
assert flickr_photo_id('10005059_b9e2a7a1ab_1013_58882012@N00.jpg') == '10005059'  # Im2GPS3k
assert flickr_photo_id('d0_b2_10005059.jpg') == '10005059'                         # MP-16 flat
assert flickr_photo_id('d0/b2/10005059.jpg') == '10005059'                         # MP-16 sharded
assert flickr_photo_id('/abs/benchmarks/im2gps3k/0010005059.jpg') == '10005059'    # leading zeros
assert flickr_photo_id('8217e6ef6f5de9d1a9e6.jpg') is None                         # hash name
assert flickr_owner('10005059_b9e2a7a1ab_1013_58882012@N00.jpg') == '58882012@n00'
assert flickr_owner('d0_b2_10005059.jpg') is None
print('parsing ok')

# --- geo gate: great-circle radius, across the antimeridian ----------------
bench = np.array([[48.8566, 2.3522], [0.0, 179.999]])
train = np.array([[48.8566, 2.3522 + 0.012],   # ~0.88 km east of Paris point
                  [48.8566, 2.3522 + 0.016],   # ~1.17 km
                  [0.0, -179.995],             # ~0.67 km, other side of 180
                  [10.0, 10.0]])
assert list(geo_candidates(train, bench, 1.0)) == [0, 2]
assert list(geo_candidates(train, bench, 2.0)) == [0, 1, 2]
assert len(geo_candidates(train, bench[:0], 1.0)) == 0
print('geo gate ok')

# --- near-duplicates: scale-invariant, chunked, threshold inclusive --------
rng = np.random.default_rng(0)
b = rng.normal(size=(5, 32)).astype(np.float32)
t = rng.normal(size=(10, 32)).astype(np.float32)
t[3] = b[1] * 7.0                                  # exact dup, different scale
t[8] = b[4] + 0.05 * rng.normal(size=32)           # near dup
ti, bi, s = near_duplicates(t, b, 0.95, chunk=4)
assert sorted(zip(ti.tolist(), bi.tolist())) == [(3, 1), (8, 4)], (ti, bi, s)
assert s.min() >= 0.95
print('near duplicates ok')
