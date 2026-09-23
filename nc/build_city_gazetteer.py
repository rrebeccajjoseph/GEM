"""Builds the point gazetteer Stage E's city term looks up against: every
training row's (lat, lng, city), pooled across OSV-5M and MP-16. Unlike the
coarse raster terms (which vote a label onto a ~60km grid cell),
this stays as raw points — CandidateBank does a nearest-neighbor query
against them per fine (~0.86km) candidate, which is the resolution city
identity actually lives at.

Usage:
    python nc/build_city_gazetteer.py [--out data/energy/city_gazetteer.npz]
"""
import sys, os, argparse
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd

argp = argparse.ArgumentParser()
argp.add_argument('--out', default='data/energy/city_gazetteer.npz')
argp.add_argument('--osv5m-metadata', default='data/osv5m/metadata_osv5m.csv')
argp.add_argument('--mp16-metadata', default='data/mp16/metadata_mp16.csv')
args = argp.parse_args()

frames = []
osv = pd.read_csv(args.osv5m_metadata, dtype={'id': str})
frames.append(osv[['lat', 'lng', 'city']])
if os.path.exists(args.mp16_metadata):
    mp16 = pd.read_csv(args.mp16_metadata, dtype={'id': str})
    frames.append(mp16[['lat', 'lng', 'city']])
else:
    print(f'{args.mp16_metadata} not found — OSV-5M only.')
pooled = pd.concat(frames, ignore_index=True)
valid = pooled['city'].notna()
pooled = pooled.loc[valid]

# No cross-dataset name harmonization attempted (same caveat as region: the
# two sources may name the same real city differently) — a nearest-point
# lookup still lands on a genuinely nearby labeled point either way, unlike
# the coarse rasters' majority vote, which this fragmentation would corrupt
# more directly.
codes, categories = pd.factorize(pooled['city'].values)
np.savez(args.out, lat=pooled['lat'].values.astype(np.float64),
        lng=pooled['lng'].values.astype(np.float64),
        city_code=codes.astype(np.int32),
        categories=np.array(categories, dtype=object))
print(f'Wrote {args.out}: {len(pooled)} points, {len(categories)} distinct cities.')
