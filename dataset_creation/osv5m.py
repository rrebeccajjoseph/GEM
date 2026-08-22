"""OSV-5M adapter.

Downloads OpenStreetView-5M (Astruc et al., CVPR 2024) from HuggingFace and
produces a metadata CSV in the format the PIGEOTTO (--yfcc) pipeline expects,
so both the PIGEON retrain and the energy model consume the same data spine.

Usage (run from the repository root):

    python -m dataset_creation.osv5m download [--split all|train|test] [--keep-zips]
    python -m dataset_creation.osv5m adapt    [--val-size 10000]
    python -m dataset_creation.osv5m augment  [--chunk-size 200000]
    python -m dataset_creation.osv5m all

`download` fetches train.csv/test.csv and the image zip shards, extracting them
under IMAGE_PATH_OSV. `adapt` builds the base metadata CSV (image, lat, lng,
month, selection + passthrough columns). `augment` runs GeoAugmentor in
resumable chunks, fills raster gaps, constructs the scaled *_reg regression
targets, and fits/saves the sklearn scaler that evaluation/metrics.py inverts.

The final CSV satisfies dataset_creation/finetune/finetune_dataset.py:
a single `image` column (path relative to IMAGE_PATH_OSV), `lat`, `lng`,
`selection` in {train, val, test}, `month` (0-11, -100 = missing, which
nn.CrossEntropyLoss ignores by default), `climate_zone` (0-29), `geo_area`,
and the six `*_reg` columns. `geo_area` being present means
generate_finetune_dataset never triggers its inline GeoAugmentor call.
"""

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)

import glob
import zipfile
import logging
import argparse
import numpy as np
import pandas as pd
from config import OSV5M_HF_REPO, OSV5M_ROOT, METADATA_PATH_OSV, \
                   IMAGE_PATH_OSV, SCALER_PATH_OSV

logger = logging.getLogger('osv5m')
logging.basicConfig(level=logging.INFO)

RAW_DIR = os.path.join(OSV5M_ROOT, 'raw')
CHUNK_DIR = os.path.join(OSV5M_ROOT, 'aug_chunks')

# Order must match evaluation/metrics.py recover_regression_values
REG_COLUMNS = ['elevation', 'population', 'temp_avg', 'temp_diff', 'prec_avg', 'prec_diff']

# Offsets applied before the log transform; 416 matches the YFCC branch of
# recover_regression_values, which the OSV path reuses (offset_val = 416).
REG_OFFSETS = np.array([416.0, 1.0, 0.0, 1.0, 1.0, 1.0])

# All columns except temp_avg (index 2) are log-transformed
LOG_COLUMN_MASK = np.array([True, True, False, True, True, True])


def download(split: str='all', keep_zips: bool=False) -> None:
    """Downloads OSV-5M metadata and image shards, extracting zips.

    Args:
        split (str, optional): which image split to fetch ('train', 'test', 'all').
        keep_zips (bool, optional): whether to keep zip shards after extraction.
    """
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
    """Walks IMAGE_PATH_OSV and maps image id (filename stem) to relative path.

    Returns:
        dict: image id -> path relative to IMAGE_PATH_OSV
    """
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
    """Builds the base PIGEON-format metadata CSV from OSV-5M's train/test CSVs.

    Args:
        val_size (int, optional): rows carved out of train as the val split.
        seed (int, optional): RNG seed for the val carve (330 = repo convention).
    """
    image_index = _index_images()
    frames = []

    for split in ['train', 'test']:
        csv_path = os.path.join(RAW_DIR, f'{split}.csv')
        logger.info(f'Reading {csv_path}.')
        df = pd.read_csv(csv_path, dtype={'id': str})
        df = df.rename(columns={'latitude': 'lat', 'longitude': 'lng'})

        # Month from capture timestamp (ms epoch); -100 = CrossEntropyLoss ignore_index
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

    # Carve a val split out of train
    train_idx = data.index[data['selection'] == 'train']
    rng = np.random.default_rng(seed)
    val_idx = rng.choice(train_idx, size=min(val_size, len(train_idx)), replace=False)
    data.loc[val_idx, 'selection'] = 'val'

    os.makedirs(os.path.dirname(METADATA_PATH_OSV), exist_ok=True)
    data.to_csv(METADATA_PATH_OSV, index=False)
    logger.info(f'Wrote {len(data)} rows to {METADATA_PATH_OSV} '
                f'({dict(data["selection"].value_counts())}).')


def _fill_nearest(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Fills NaNs in a column with the value of the nearest valid row (lat/lng).

    Args:
        data (pd.DataFrame): dataframe with lat/lng columns
        column (str): column whose NaNs to fill

    Returns:
        pd.DataFrame: dataframe with NaNs filled
    """
    from scipy.spatial import cKDTree

    invalid = data[column].isnull()
    if invalid.sum() == 0:
        return data

    logger.warning(f'Filling {invalid.sum()} missing {column} values via nearest neighbor.')
    valid = data[~invalid]
    tree = cKDTree(valid[['lat', 'lng']].values)
    _, nearest = tree.query(data.loc[invalid, ['lat', 'lng']].values, k=1)
    data.loc[invalid, column] = valid.iloc[nearest][column].values
    return data


def augment(chunk_size: int=200000) -> None:
    """Runs GeoAugmentor over the metadata CSV in resumable chunks, then
    fills raster gaps, builds *_reg columns, and saves the regression scaler.

    Args:
        chunk_size (int, optional): rows per GeoAugmentor chunk.
    """
    import joblib
    from sklearn.preprocessing import StandardScaler
    from preprocessing import GeoAugmentor

    data = pd.read_csv(METADATA_PATH_OSV, dtype={'id': str})
    os.makedirs(CHUNK_DIR, exist_ok=True)
    os.makedirs('data/tmp', exist_ok=True)  # GeoAugmentor checkpoint location

    # Chunked augmentation with resume: a chunk file with the right row count is done
    n_chunks = int(np.ceil(len(data) / chunk_size))
    chunks = []
    for i in range(n_chunks):
        chunk_path = os.path.join(CHUNK_DIR, f'chunk_{i:04d}.csv')
        chunk = data.iloc[i * chunk_size:(i + 1) * chunk_size]

        if os.path.exists(chunk_path):
            done = pd.read_csv(chunk_path, dtype={'id': str})
            if len(done) == len(chunk) and 'climate_zone' in done.columns:
                logger.info(f'Chunk {i + 1}/{n_chunks} already augmented, skipping.')
                chunks.append(done)
                continue

        logger.info(f'Augmenting chunk {i + 1}/{n_chunks} ({len(chunk)} rows).')
        augmentor = GeoAugmentor(output_file=chunk_path)
        chunk = augmentor(chunk.reset_index(drop=True))
        chunk.to_csv(chunk_path, index=False)
        chunks.append(chunk)

    data = pd.concat(chunks, ignore_index=True)

    # Raster gaps: ocean-adjacent Koppen pixels and SRTM's +/-60 degree coverage
    data = _fill_nearest(data, 'climate_zone')
    data['climate_zone'] = data['climate_zone'].astype(int)

    n_elev = data['elevation'].isnull().sum()
    if n_elev > 0:
        logger.warning(f'Filling {n_elev} missing elevation values with 0 (SRTM coverage).')
        data['elevation'] = data['elevation'].fillna(0)

    data['population'] = data['population'].fillna(0).clip(lower=0)

    # Scaled regression targets, matching recover_regression_values' inverse:
    # value + offset -> log (all but temp_avg) -> StandardScaler (fit on train)
    raw = data[REG_COLUMNS].values.astype(np.float64) + REG_OFFSETS

    clipped = (raw[:, LOG_COLUMN_MASK] < 0.1).sum()
    if clipped > 0:
        logger.warning(f'Clipping {clipped} non-positive values before log transform.')
    raw[:, LOG_COLUMN_MASK] = np.log(np.maximum(raw[:, LOG_COLUMN_MASK], 0.1))

    scaler = StandardScaler()
    train_mask = (data['selection'] == 'train').values
    scaler.fit(raw[train_mask])
    scaled = scaler.transform(raw)

    for j, col in enumerate(REG_COLUMNS):
        data[f'{col}_reg'] = scaled[:, j]

    os.makedirs(os.path.dirname(SCALER_PATH_OSV), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH_OSV)
    logger.info(f'Saved regression scaler to {SCALER_PATH_OSV}.')

    data.to_csv(METADATA_PATH_OSV, index=False)
    logger.info(f'Wrote augmented metadata ({len(data)} rows) to {METADATA_PATH_OSV}.')


def main():
    argp = argparse.ArgumentParser(description='OSV-5M download and adaptation.')
    argp.add_argument('command', choices=['download', 'adapt', 'augment', 'all'])
    argp.add_argument('--split', choices=['train', 'test', 'all'], default='all',
                      help='Image split to download.')
    argp.add_argument('--keep-zips', action='store_true', default=False,
                      help='Keep zip shards after extraction.')
    argp.add_argument('--val-size', type=int, default=10000,
                      help='Rows carved out of train as the val split.')
    argp.add_argument('--chunk-size', type=int, default=200000,
                      help='Rows per GeoAugmentor chunk.')
    args = argp.parse_args()

    if args.command in ('download', 'all'):
        download(split=args.split, keep_zips=args.keep_zips)

    if args.command in ('adapt', 'all'):
        adapt(val_size=args.val_size)

    if args.command in ('augment', 'all'):
        augment(chunk_size=args.chunk_size)


if __name__ == '__main__':
    main()
