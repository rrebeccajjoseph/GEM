"""OSV-5M adapter (standalone).

Downloads OpenStreetView-5M (Astruc et al., CVPR 2024) from HuggingFace and
builds the metadata CSV the energy pipeline consumes: image path, lat, lng,
month, selection, climate_zone, drive_side.

Usage (from the repository root):

    python osv5m.py download [--split all|train|test] [--keep-zips]
    python osv5m.py adapt    [--val-size 10000]
    python osv5m.py climate                       # needs get_rasters.sh first
    python osv5m.py all

`climate` samples the Köppen-Geiger raster at each point (classes 0-29, ocean
pixels filled from the nearest valid point) — used by the season head and the
mask-informativeness analysis. The PIGEON-baseline retrain needs richer
auxiliary labels (elevation etc.); that pipeline lives in a separate clone of
the official PIGEON release, not here.
"""

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

import glob
import zipfile
import logging
import argparse
import numpy as np
import pandas as pd
from config import OSV5M_HF_REPO, OSV5M_ROOT, METADATA_PATH_OSV, \
                   IMAGE_PATH_OSV, KOPPEN_GEIGER_PATH

logger = logging.getLogger('osv5m')
logging.basicConfig(level=logging.INFO)

RAW_DIR = os.path.join(OSV5M_ROOT, 'raw')


def download(split: str='all', keep_zips: bool=False) -> None:
    """Downloads OSV-5M metadata and image shards, extracting zips."""
    from huggingface_hub import snapshot_download

    patterns = ['train.csv', 'test.csv']
    splits = ['train', 'test'] if split == 'all' else [split]
    patterns += [f'images/{s}/*' for s in splits]

    logger.info(f'Downloading {OSV5M_HF_REPO} ({split}) to {RAW_DIR}.')
    snapshot_download(repo_id=OSV5M_HF_REPO, repo_type='dataset',
                      local_dir=RAW_DIR, allow_patterns=patterns)

    for s in splits:
        zip_paths = sorted(glob.glob(os.path.join(RAW_DIR, 'images', s, '*.zip')))
        out_dir = os.path.join(IMAGE_PATH_OSV, s)
        os.makedirs(out_dir, exist_ok=True)

        for zip_path in zip_paths:
            marker = zip_path + '.extracted'
            if os.path.exists(marker):
                continue

            logger.info(f'Extracting {zip_path} -> {out_dir}')
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(out_dir)

            with open(marker, 'w') as f:
                f.write('ok')

            if not keep_zips:
                os.remove(zip_path)


def _index_images() -> dict:
    """Maps image id (filename stem) -> path relative to IMAGE_PATH_OSV."""
    logger.info(f'Indexing images under {IMAGE_PATH_OSV} (one-time walk).')
    index = {}
    for root, _, files in os.walk(IMAGE_PATH_OSV):
        for name in files:
            stem, ext = os.path.splitext(name)
            if ext.lower() in ('.jpg', '.jpeg', '.png'):
                rel = os.path.relpath(os.path.join(root, name), IMAGE_PATH_OSV)
                index[stem] = rel

    logger.info(f'Indexed {len(index)} images.')
    return index


def adapt(val_size: int=10000, seed: int=330) -> None:
    """Builds the base metadata CSV from OSV-5M's train/test CSVs."""
    image_index = _index_images()
    frames = []

    for split in ['train', 'test']:
        csv_path = os.path.join(RAW_DIR, f'{split}.csv')
        logger.info(f'Reading {csv_path}.')
        df = pd.read_csv(csv_path, dtype={'id': str})
        df = df.rename(columns={'latitude': 'lat', 'longitude': 'lng'})

        # Month from capture timestamp (ms epoch); -100 = missing
        captured = pd.to_datetime(df['captured_at'], unit='ms', errors='coerce')
        df['month'] = (captured.dt.month - 1).fillna(-100).astype(int)

        df['image'] = df['id'].map(image_index)
        missing = df['image'].isnull().sum()
        if missing > 0:
            logger.warning(f'{split}: dropping {missing} rows with no image file on disk.')
            df = df[df['image'].notnull()]

        df['selection'] = split
        frames.append(df[['image', 'lat', 'lng', 'month', 'selection',
                          'id', 'country', 'drive_side']])

    data = pd.concat(frames, ignore_index=True)

    train_idx = data.index[data['selection'] == 'train']
    rng = np.random.default_rng(seed)
    val_idx = rng.choice(train_idx, size=min(val_size, len(train_idx)), replace=False)
    data.loc[val_idx, 'selection'] = 'val'

    os.makedirs(os.path.dirname(METADATA_PATH_OSV), exist_ok=True)
    data.to_csv(METADATA_PATH_OSV, index=False)
    logger.info(f'Wrote {len(data)} rows to {METADATA_PATH_OSV} '
                f'({dict(data["selection"].value_counts())}).')


def climate(batch_size: int=500000) -> None:
    """Adds climate_zone (Köppen classes 0-29) by sampling the raster at
    each point; ocean/nodata pixels are filled from the nearest valid row."""
    from scipy.spatial import cKDTree
    from energy.grid import RasterioSampler

    data = pd.read_csv(METADATA_PATH_OSV, dtype={'id': str})
    sampler = RasterioSampler(KOPPEN_GEIGER_PATH, nodata=0,
                              transform_value=lambda v: v - 1)

    vals = np.empty(len(data))
    for start in range(0, len(data), batch_size):
        chunk = data.iloc[start:start + batch_size]
        vals[start:start + len(chunk)] = sampler(chunk['lat'].values,
                                                 chunk['lng'].values)

    invalid = np.isnan(vals)
    if invalid.any():
        logger.warning(f'Filling {invalid.sum()} ocean/nodata climate values '
                       f'via nearest neighbor.')
        tree = cKDTree(data.loc[~invalid, ['lat', 'lng']].values)
        _, nearest = tree.query(data.loc[invalid, ['lat', 'lng']].values, k=1)
        vals[invalid] = vals[~invalid][nearest]

    data['climate_zone'] = vals.astype(int)
    data.to_csv(METADATA_PATH_OSV, index=False)
    logger.info(f'Wrote climate_zone to {METADATA_PATH_OSV}.')


def main():
    argp = argparse.ArgumentParser(description='OSV-5M download and adaptation.')
    argp.add_argument('command', choices=['download', 'adapt', 'climate', 'all'])
    argp.add_argument('--split', choices=['train', 'test', 'all'], default='all')
    argp.add_argument('--keep-zips', action='store_true', default=False)
    argp.add_argument('--val-size', type=int, default=10000)
    args = argp.parse_args()

    if args.command in ('download', 'all'):
        download(split=args.split, keep_zips=args.keep_zips)
    if args.command in ('adapt', 'all'):
        adapt(val_size=args.val_size)
    if args.command in ('climate', 'all'):
        climate()


if __name__ == '__main__':
    main()
