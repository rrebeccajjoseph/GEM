"""MP-16-Pro adapter (standalone).

Downloads MP16-Pro (Jia et al., NeurIPS 2024 "G3") from HuggingFace — a
broad Flickr photo corpus (landmarks, indoor scenes, people, food: the same
kind of imagery the Im2GPS/YFCC benchmarks draw from, unlike OSV-5M's
street-level dashcam footage) — and builds the metadata CSV the energy
pipeline consumes: image filename, lat, lng, selection.

Unlike OSV-5M's independent per-split zips, MP16-Pro ships as 19 raw
byte-split chunks of ONE tar archive (the dataset card's own instructions
are literally `cat mp-16-images* > mp-16-images.tar`) — extraction is a
single streamed pass over all 19 chunks in tar-entry order and is not
incrementally resumable. A kill mid-extract means re-running `extract`
from scratch: tens of minutes on local NVMe, not hours, so this is left
unoptimized rather than adding checkpoint complexity for a cheap redo.

Usage (from the repository root):

    python mp16.py download
    python mp16.py extract
    python mp16.py adapt    [--val-frac 0.01]
    python mp16.py all
"""

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

import glob
import subprocess
import time
import logging
import argparse
import numpy as np
import pandas as pd
from val_split import spatial_val_split
from config import MP16_HF_REPO, MP16_ROOT, METADATA_PATH_MP16, IMAGE_PATH_MP16

logger = logging.getLogger('mp16')
logging.basicConfig(level=logging.INFO)

RAW_DIR = os.path.join(MP16_ROOT, 'raw')
FILTERED_CSV = os.path.join(RAW_DIR, 'metadata', 'MP16_Pro_filtered.csv')


def download() -> None:
    """Downloads the 19 tar shards + metadata CSVs. Gated: the account whose
    token lives on this machine (~/.cache/huggingface/token or $HF_TOKEN)
    must already have clicked "Agree and access" on the dataset page.

    huggingface_hub looks for the token under $HF_HOME (default
    ~/.cache/huggingface), not always ~/.cache/huggingface itself — nc/lib.sh
    exports HF_HOME to a project-local cache dir for CLIP model downloads,
    which silently redirects token lookup there too and made every call in
    this function 401 as an anonymous request, even though the exact same
    call from a shell that never sourced lib.sh worked fine. Reading the
    token explicitly from its real path sidesteps whatever HF_HOME the
    calling shell happens to have set."""
    from huggingface_hub import hf_hub_download

    token_path = os.path.expanduser('~/.cache/huggingface/token')
    token = open(token_path).read().strip() if os.path.exists(token_path) else None
    if token is None:
        logger.warning(f'No token at {token_path}; falling back to ambient auth '
                       f'(HF_TOKEN env var or default lookup) — likely to 401 on a '
                       f'gated repo if that is also empty.')

    wanted = [f'mp-16-images{i:02d}' for i in range(19)] + [
        'metadata/mp16_urls.csv', 'metadata/MP16_Pro_places365.csv',
        'metadata/MP16_Pro_filtered.csv', 'metadata/tar_index.pkl']
    logger.info(f'Downloading {len(wanted)} files from {MP16_HF_REPO} to {RAW_DIR} (~370 GB).')
    for i, f in enumerate(wanted):
        dest = os.path.join(RAW_DIR, f)
        if os.path.exists(dest):
            logger.info(f'[{i + 1}/{len(wanted)}] {f} already present, skip.')
            continue
        logger.info(f'[{i + 1}/{len(wanted)}] {f}')
        for attempt in range(3):
            try:
                hf_hub_download(repo_id=MP16_HF_REPO, repo_type='dataset',
                                filename=f, local_dir=RAW_DIR, token=token)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                logger.warning(f'{f}: attempt {attempt + 1} failed ({e}); retrying.')
                time.sleep(5)
    logger.info('Download complete.')


def extract() -> None:
    """Streams all 19 shards through `tar x` as one continuous archive (they
    are a raw byte split of a single tar, not independent per-chunk
    archives — piping `cat` into `tar` avoids materializing the ~370 GB
    concatenated file). Entries are `./images/<file>.jpg`, so extracting
    into MP16_ROOT reproduces `MP16_ROOT/images/...` == IMAGE_PATH_MP16."""
    marker = os.path.join(RAW_DIR, '.extracted')
    if os.path.exists(marker):
        logger.info('Already extracted (found .extracted marker).')
        return
    shards = sorted(glob.glob(os.path.join(RAW_DIR, 'mp-16-images*')))
    assert shards, f'No mp-16-images* shards found under {RAW_DIR}; run download first.'
    total_gb = sum(os.path.getsize(s) for s in shards) / 1e9
    logger.info(f'Extracting {len(shards)} shards ({total_gb:.0f} GB) -> {IMAGE_PATH_MP16}')
    os.makedirs(MP16_ROOT, exist_ok=True)
    cat = subprocess.Popen(['cat'] + shards, stdout=subprocess.PIPE)
    subprocess.run(['tar', 'xf', '-', '-C', MP16_ROOT], stdin=cat.stdout, check=True)
    cat.stdout.close()
    cat.wait()
    with open(marker, 'w') as fh:
        fh.write('ok')
    logger.info('Extraction complete.')


def adapt(val_frac: float=0.01, seed: int=330) -> None:
    """Builds the metadata CSV from MP16_Pro_filtered.csv, keeping only rows
    whose image actually extracted (some Flickr URLs had already rotted at
    collection time — the dataset card's own gap between 4.65M and ~4.12M)."""
    logger.info(f'Reading {FILTERED_CSV}.')
    df = pd.read_csv(FILTERED_CSV, dtype={'IMG_ID': str})
    df = df.rename(columns={'IMG_ID': 'image', 'LAT': 'lat', 'LON': 'lng'})
    # country/region: already columns in MP16_Pro_filtered.csv, no geocoding
    # needed — previously dropped here despite costing nothing to keep.
    # Using country_code (lowercase ISO alpha-2, e.g. 'us'), not the 'country'
    # column (full English names, e.g. 'United States') — OSV-5M's own
    # 'country' column is uppercase ISO alpha-2, and pooling the two without
    # this would silently split every real country into two label classes
    # (confirmed: 446 "countries" came out of the first pooled build, not the
    # true ~195-225 — fixed here rather than downstream at pool time so the
    # metadata CSV itself carries a comparable code).
    # S365_Label (Places365 scene category) and Prob_indoor/natural/urban:
    # also already columns here — a data-driven scene/propensity raster
    # like climate_zone, not an external source.
    # city: for Stage E's fine-candidate city-compatibility term (a
    # nearest-observation lookup, not a coarse grid raster).
    df = df[['image', 'lat', 'lng', 'country_code', 'region', 'S365_Label',
             'Prob_indoor', 'Prob_natural', 'Prob_urban', 'city']].dropna(subset=['lat', 'lng'])
    df = df.rename(columns={'country_code': 'country', 'S365_Label': 'scene'})
    df['country'] = df['country'].str.upper()

    logger.info(f'Indexing images under {IMAGE_PATH_MP16} (one-time walk).')
    present = {entry.name for entry in os.scandir(IMAGE_PATH_MP16)}
    missing = (~df['image'].isin(present)).sum()
    if missing:
        logger.warning(f'Dropping {missing} rows with no extracted image file.')
        df = df[df['image'].isin(present)]
    df['id'] = df['image'].str.rsplit('.', n=1).str[0]

    df['selection'] = 'train'
    # spatially held out: Flickr bursts leak across a random split (see val_split.py)
    df['selection'] = spatial_val_split(df, int(len(df) * val_frac), seed=seed)

    os.makedirs(os.path.dirname(METADATA_PATH_MP16), exist_ok=True)
    df[['image', 'lat', 'lng', 'selection', 'id', 'country', 'region', 'scene',
        'Prob_indoor', 'Prob_natural', 'Prob_urban', 'city']].to_csv(
        METADATA_PATH_MP16, index=False)
    logger.info(f'Wrote {len(df)} rows to {METADATA_PATH_MP16} '
                f'({dict(df["selection"].value_counts())}).')


def main():
    argp = argparse.ArgumentParser(description='MP-16-Pro adapter.')
    argp.add_argument('command', choices=['download', 'extract', 'adapt', 'all'])
    argp.add_argument('--val-frac', type=float, default=0.01)
    args = argp.parse_args()

    if args.command in ('download', 'all'):
        download()
    if args.command in ('extract', 'all'):
        extract()
    if args.command in ('adapt', 'all'):
        adapt(val_frac=args.val_frac)


if __name__ == '__main__':
    main()
