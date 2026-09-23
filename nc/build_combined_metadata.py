"""Merges OSV-5M and MP-16-Pro metadata into one CSV finetune_encoder can read
with --images '' (each 'image' path is already relative to the repo root, so
os.path.join('', p) == p). Only the columns finetune_encoder needs: id, lat,
lng, image, selection."""
import sys, os
sys.path.insert(0, os.getcwd())
import pandas as pd
from config import METADATA_PATH_OSV, IMAGE_PATH_OSV, METADATA_PATH_MP16, IMAGE_PATH_MP16

osv = pd.read_csv(METADATA_PATH_OSV, dtype={'id': str})
osv = osv[osv['selection'].isin(['train', 'val'])][['id', 'lat', 'lng', 'image', 'selection']].copy()
osv['image'] = IMAGE_PATH_OSV + '/' + osv['image']

mp16 = pd.read_csv(METADATA_PATH_MP16, dtype={'id': str})
mp16 = mp16[['id', 'lat', 'lng', 'image', 'selection']].copy()
mp16['id'] = 'mp16_' + mp16['id']  # osv5m ids are bare numeric strings; avoid collisions
mp16['image'] = IMAGE_PATH_MP16 + '/' + mp16['image']

combined = pd.concat([osv, mp16], ignore_index=True)
out = 'data/combined_osv5m_mp16_metadata.csv'
combined.to_csv(out, index=False)
print(f'Wrote {len(combined)} rows to {out} '
      f'(osv5m {len(osv)}, mp16 {len(mp16)}); selection counts: '
      f'{dict(combined["selection"].value_counts())}')
