"""Adds every energy term collected but sitting unused in OSV-5M's and
MP-16-Pro's own metadata — no geocoding, no external raster. Writes a NEW
grid file rather than overwriting data/energy/grid.npz in place — the
in-flight mp16_full pipeline's later Stage D step still reads the original,
so this keeps that run a clean ablation instead of silently changing what
--rasters means underneath it. Point --rasters --grid at the new file
explicitly to use these terms; the original is untouched either way.

Terms built, and why each is categorical (mode, aggregate_labels_over_cells)
or continuous (mean, aggregate_values_over_cells):
  country, region     — both datasets; categorical.
  subregion           — OSV-5M only (finer than region); categorical.
  land_cover, soil    — OSV-5M only; small fixed code sets; categorical.
  road_index          — OSV-5M only; stored as float but only ~8 distinct
                        values observed — a small mode-aggregated category
                        fits it better than a Gaussian regression head.
  dist_sea            — OSV-5M only; genuinely continuous (distance to
                        coastline); mean-aggregated, GaussianHead.
  scene               — MP-16 only (Places365 S365_Label, 365 classes);
                        categorical, the classic im2gps-era scene signal.
  prob_indoor/natural/urban — MP-16 only; continuous propensities.

Explicitly NOT built: city (both datasets have it, but even at 500k OSV-5M
rows it's already 4,000+ categories with a 39-row median and an 11-row 25th
percentile — at H3 res-4 a single ~60km cell often spans a whole metro area
anyway, so the head would need tens of thousands of classes for signal
region/country mostly already carry) and continent (MP-16's column header
exists but is entirely empty — 0 non-null across all 4.12M rows, checked
directly rather than assumed).

Usage:
    python nc/build_geo_rasters.py [--out data/energy/grid_geo.npz]
"""
import sys, os, argparse
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
from energy.grid import (build_grid, aggregate_labels_over_cells,
                         aggregate_values_over_cells)

argp = argparse.ArgumentParser()
argp.add_argument('--grid', default='data/energy/grid.npz')
argp.add_argument('--out', default='data/energy/grid_geo.npz')
argp.add_argument('--osv5m-metadata', default='data/osv5m/metadata_osv5m.csv')
argp.add_argument('--mp16-metadata', default='data/mp16/metadata_mp16.csv')
args = argp.parse_args()

base = dict(np.load(args.grid, allow_pickle=False))
resolution = int(base['resolution'])
cells, _ = build_grid(resolution)

osv = pd.read_csv(args.osv5m_metadata, dtype={'id': str})
have_mp16 = os.path.exists(args.mp16_metadata)
mp16 = pd.read_csv(args.mp16_metadata, dtype={'id': str}) if have_mp16 else None
if not have_mp16:
    print(f'{args.mp16_metadata} not found — MP-16-only terms (scene, '
          f'prob_indoor/natural/urban) will be skipped.')

# country/region: pool both datasets for one consistent category set.
pooled_country_region = pd.concat(
    [osv[['lat', 'lng', 'country', 'region']]]
    + ([mp16[['lat', 'lng', 'country', 'region']]] if have_mp16 else []),
    ignore_index=True)
# Defensive, in addition to each adapter using a comparable code: country is
# an ISO alpha-2 code in both sources but the case wasn't consistent until
# mp16.py's adapt() was fixed (2026-09-08) — normalizing here too means this
# script can't silently reintroduce the same duplicate-class bug (446
# "countries" instead of ~200) if a metadata CSV predates that fix.
pooled_country_region['country'] = pooled_country_region['country'].str.upper()
print(f'Pooled {len(pooled_country_region)} rows for country/region '
      f'(country non-null: {pooled_country_region.country.notna().sum()}, '
      f'region non-null: {pooled_country_region.region.notna().sum()}).')

categorical_specs = [
    # (raster name, dataframe, column)
    ('country', pooled_country_region, 'country'),
    ('region', pooled_country_region, 'region'),
    ('subregion', osv, 'subregion'),
    ('land_cover', osv, 'land_cover'),
    ('soil', osv, 'soil'),
    ('road_index', osv, 'road_index'),
]
if have_mp16:
    categorical_specs.append(('scene', mp16, 'scene'))

for name, df, col in categorical_specs:
    valid = df[col].notna()
    raster, categories = aggregate_labels_over_cells(
        cells, resolution, df.loc[valid, 'lat'].values,
        df.loc[valid, 'lng'].values, df.loc[valid, col].values)
    base[f'raster_{name}'] = raster
    # Saved for interpretability (code -> label), NOT prefixed raster_ —
    # load_grid() (energy/train.py) takes every raster_*-prefixed key as a
    # numeric table for RasterBank and loads with allow_pickle=False; an
    # object-dtype string array under that prefix would crash the loader
    # the moment it tried to read this key, and RasterBank would fail trying
    # to cast it to float64 even if it somehow got through.
    base[f'categories_{name}'] = np.array(categories, dtype=object)
    print(f'{name}: {len(categories)} classes, any NaN: {np.isnan(raster).any()}')

continuous_specs = [('dist_sea', osv, 'dist_sea')]
if have_mp16:
    continuous_specs += [('prob_indoor', mp16, 'Prob_indoor'),
                         ('prob_natural', mp16, 'Prob_natural'),
                         ('prob_urban', mp16, 'Prob_urban')]

for name, df, col in continuous_specs:
    valid = df[col].notna()
    raster = aggregate_values_over_cells(
        cells, resolution, df.loc[valid, 'lat'].values,
        df.loc[valid, 'lng'].values, df.loc[valid, col].values)
    base[f'raster_{name}'] = raster
    print(f'{name}: mean {raster.mean():.3f}, std {raster.std():.3f}, '
          f'any NaN: {np.isnan(raster).any()}')

np.savez(args.out, **base)
new_terms = [n for n, _, _ in categorical_specs] + [n for n, _, _ in continuous_specs]
print(f'Wrote {args.out} (original {args.grid} keys preserved, plus: '
      f'{", ".join(new_terms)}).')
