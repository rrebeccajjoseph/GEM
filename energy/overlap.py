"""Train/benchmark leakage check: finds training images that are (or nearly
are) Im2GPS3k / YFCC4k test images, before any coarse-stage run trains on them.

MP-16 and YFCC4k are both drawn from YFCC100M, and Im2GPS3k is Flickr too, so
the same photo — or the same photographer's shot of the same spot — can sit on
both sides. Three passes, cheapest first:

  1. id      Flickr photo id parsed from each filename; exact matches.
  2. author  Flickr owner (`12345678@N00`) shared by a benchmark image and a
             training image — a softer signal (same photographer, maybe the
             same trip), reported but not flagged for exclusion by default.
  3. embed   Near-duplicates. Encoding all ~4M MP-16 images would cost a GPU
             day, so only training images within --radius-km of some benchmark
             image are encoded (a re-upload keeps its geotag), then compared
             by cosine similarity of CLIP projected image embeddings.

Writes <out>/overlap_pairs.csv (every flagged pair, for eyeballing) and
<out>/exclude_ids.txt (training ids to drop: id matches + embed matches).

Usage (from the repository root, where the data lives):
    python -m energy.overlap --benchmarks im2gps3k yfcc4k \
        [--train mp16=data/mp16/metadata_mp16.csv:data/mp16/images] \
        [--mp16-raw data/mp16/raw/metadata/MP16_Pro_filtered.csv] \
        [--radius-km 1.0] [--sim 0.95] [--skip-embed]
"""

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)

import re
import json
import logging
import argparse
import numpy as np
import pandas as pd

logger = logging.getLogger('energy.overlap')
logging.basicConfig(level=logging.INFO)

EARTH_RADIUS_KM = 6371.0
OWNER_RE = re.compile(r'^\d+@N\d+$')


def flickr_photo_id(name: str):
    """Flickr photo id from an image filename, or None.

    Handles Flickr's native `photoid_secret_server_owner.jpg` (Im2GPS3k) and
    GeoEstimation's sharded `ab_cd_photoid.jpg` / `ab/cd/photoid.jpg` (MP-16):
    the photo id is the longest all-digit `_`-token of the basename. Hash-named
    files (no token of 6+ digits) return None and are left to the embed pass."""
    stem = os.path.splitext(os.path.basename(str(name)))[0]
    digits = [t for t in stem.split('_') if t.isdigit() and len(t) >= 6]
    return max(digits, key=len).lstrip('0') if digits else None


def flickr_owner(name: str):
    """Flickr owner NSID (`12345678@N00`) embedded in a native Flickr
    filename, or None."""
    stem = os.path.splitext(os.path.basename(str(name)))[0]
    owners = [t for t in stem.split('_') if OWNER_RE.match(t)]
    return owners[0].lower() if owners else None


def unit_xyz(latlngs: np.ndarray) -> np.ndarray:
    lat, lng = np.radians(latlngs[:, 0]), np.radians(latlngs[:, 1])
    return np.stack([np.cos(lat) * np.cos(lng), np.cos(lat) * np.sin(lng),
                     np.sin(lat)], axis=1)


def geo_candidates(train_latlngs: np.ndarray, bench_latlngs: np.ndarray,
                   radius_km: float) -> np.ndarray:
    """Indices of training rows within radius_km (great-circle) of any
    benchmark point — KD-tree on unit vectors with the matching chord length."""
    from scipy.spatial import cKDTree

    chord = 2 * np.sin(radius_km / (2 * EARTH_RADIUS_KM))
    tree = cKDTree(unit_xyz(train_latlngs))
    hits = tree.query_ball_point(unit_xyz(bench_latlngs), r=chord)
    return np.unique(np.concatenate([np.asarray(h, dtype=np.int64) for h in hits]
                                    or [np.zeros(0, np.int64)]))


def near_duplicates(train_emb: np.ndarray, bench_emb: np.ndarray, sim: float,
                    chunk: int=4096):
    """(train_idx, bench_idx, cosine) for every pair at or above sim. Both
    inputs are L2-normalized here, so callers can pass raw embeddings."""
    t = train_emb / np.linalg.norm(train_emb, axis=1, keepdims=True).clip(1e-8)
    b = bench_emb / np.linalg.norm(bench_emb, axis=1, keepdims=True).clip(1e-8)
    ti, bi, s = [], [], []
    for lo in range(0, len(t), chunk):
        cos = t[lo:lo + chunk] @ b.T
        r, c = np.nonzero(cos >= sim)
        ti.append(r + lo); bi.append(c); s.append(cos[r, c])
    cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)
    return cat(ti, np.int64), cat(bi, np.int64), cat(s, np.float32)


def encode_images(paths, cache_path: str=None, batch_size: int=128,
                  device: str='cpu') -> np.ndarray:
    """Projected CLIP image embeddings (the space CLIP was trained to compare
    in; the mean-pooled hidden states the pipeline trains on sit in a narrow
    cone where unrelated images already score ~0.8, useless for dedup)."""
    import torch
    from config import CLIP_MODEL
    from transformers import CLIPProcessor, CLIPVisionModelWithProjection
    from energy.embed_cache import ImageDataset

    if cache_path and os.path.exists(cache_path):
        emb = np.load(cache_path)
        if emb.shape[0] == len(paths):
            logger.info(f'Loaded cached embeddings: {cache_path}.')
            return emb

    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL).to(device).eval()
    loader = torch.utils.data.DataLoader(ImageDataset(list(paths), processor),
                                         batch_size=batch_size, num_workers=8)
    out, row = None, 0
    with torch.no_grad():
        for pixels in loader:
            e = model(pixel_values=pixels.to(device)).image_embeds.float().cpu().numpy()
            if out is None:
                out = np.zeros((len(paths), e.shape[1]), dtype=np.float32)
            out[row:row + len(e)] = e
            row += len(e)
            if row % (batch_size * 50) < batch_size:
                logger.info(f'{row}/{len(paths)} encoded.')
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, out)
    return out


def parse_train_spec(spec: str):
    """`name=meta.csv:images_dir` -> (name, meta, images)."""
    name, rest = spec.split('=', 1)
    meta, images = rest.rsplit(':', 1)
    return name, meta, images


def main():
    from config import METADATA_PATH_MP16, IMAGE_PATH_MP16
    from energy.benchmark import load_benchmark
    from energy.embed_cache import pick_device

    argp = argparse.ArgumentParser(description='Train/benchmark leakage check.')
    argp.add_argument('--benchmarks', nargs='+', default=['im2gps3k', 'yfcc4k'])
    argp.add_argument('--train', nargs='+',
                      default=[f'mp16={METADATA_PATH_MP16}:{IMAGE_PATH_MP16}'],
                      help='name=meta.csv:images_dir, one per training source. The CSV '
                           'needs id, image, lat, lng (the adapters\' output format).')
    argp.add_argument('--mp16-raw', default='data/mp16/raw/metadata/MP16_Pro_filtered.csv',
                      help='MP-16 source CSV, for its AUTHOR column (author pass).')
    argp.add_argument('--radius-km', type=float, default=1.0)
    argp.add_argument('--sim', type=float, default=0.95)
    argp.add_argument('--max-candidates', type=int, default=500_000,
                      help='Abort the embed pass above this many geo candidates '
                           '(lower --radius-km) rather than silently run for a day.')
    argp.add_argument('--skip-embed', action='store_true')
    argp.add_argument('--out', default='data/benchmarks/overlap')
    args = argp.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = pick_device()

    bench = []
    for name in args.benchmarks:
        paths, latlngs = load_benchmark(name)
        bench.append(pd.DataFrame({'benchmark': name, 'bench_image': paths,
                                   'bench_lat': latlngs[:, 0], 'bench_lng': latlngs[:, 1]}))
    bench = pd.concat(bench, ignore_index=True)
    bench['photo_id'] = bench['bench_image'].map(flickr_photo_id)
    bench['owner'] = bench['bench_image'].map(flickr_owner)
    for name, g in bench.groupby('benchmark'):
        logger.info(f'{name}: {len(g)} images, {g["photo_id"].notna().sum()} with a '
                    f'parsable Flickr photo id, {g["owner"].notna().sum()} with an owner '
                    f'(e.g. {os.path.basename(g["bench_image"].iloc[0])}).')

    pairs, summary = [], {}
    for spec in args.train:
        src, meta_path, image_dir = parse_train_spec(spec)
        train = pd.read_csv(meta_path, dtype={'id': str, 'image': str})
        train['photo_id'] = train['image'].map(flickr_photo_id)
        logger.info(f'{src}: {len(train)} rows, {train["photo_id"].notna().sum()} with a '
                    f'parsable Flickr photo id (e.g. {train["image"].iloc[0]}).')
        s = summary.setdefault(src, {})

        # 1. id
        m = bench.dropna(subset=['photo_id']).merge(
            train.dropna(subset=['photo_id'])[['id', 'image', 'lat', 'lng', 'photo_id']],
            on='photo_id')
        s['id'] = m.groupby('benchmark').size().to_dict()
        pairs.append(m.assign(source=src, reason='id', sim=np.nan))

        # 2. author (MP-16 only: its raw CSV carries AUTHOR)
        if src == 'mp16' and os.path.exists(args.mp16_raw):
            raw = pd.read_csv(args.mp16_raw, usecols=lambda c: c in ('IMG_ID', 'AUTHOR'),
                              dtype=str)
            if 'AUTHOR' in raw.columns:
                owners = set(raw['AUTHOR'].dropna().str.lower())
                hit = bench['owner'].isin(owners)
                s['author_shared_bench_images'] = bench[hit].groupby('benchmark').size().to_dict()
            else:
                logger.warning(f'{args.mp16_raw} has no AUTHOR column; skipping author pass.')

        # 3. embed
        if args.skip_embed:
            continue
        cand = geo_candidates(train[['lat', 'lng']].values.astype(np.float64),
                              bench[['bench_lat', 'bench_lng']].values.astype(np.float64),
                              args.radius_km)
        logger.info(f'{src}: {len(cand)} training images within {args.radius_km} km of a '
                    f'benchmark image.')
        s['geo_candidates'] = int(len(cand))
        if len(cand) > args.max_candidates:
            raise SystemExit(f'{len(cand)} candidates > --max-candidates '
                             f'{args.max_candidates}; lower --radius-km.')
        bench_emb = encode_images(bench['bench_image'].values,
                                  os.path.join(args.out, 'bench_emb.npy'), device=device)
        cand_rows = train.iloc[cand]
        cand_emb = encode_images([os.path.join(image_dir, p) for p in cand_rows['image']],
                                 os.path.join(args.out, f'{src}_cand_emb_{args.radius_km}km.npy'),
                                 device=device)
        ti, bi, sim = near_duplicates(cand_emb, bench_emb, args.sim)
        e = pd.concat([bench.iloc[bi].reset_index(drop=True),
                       cand_rows.iloc[ti][['id', 'image', 'lat', 'lng']].reset_index(drop=True)],
                      axis=1).assign(source=src, reason='embed', sim=sim)
        s['embed'] = e.groupby('benchmark').size().to_dict()
        pairs.append(e)
        # Top-1 similarity per benchmark image, to judge the threshold.
        top1 = np.full(len(bench), -1.0, dtype=np.float32)
        _, bi_all, sim_all = near_duplicates(cand_emb, bench_emb, 0.5)
        np.maximum.at(top1, bi_all, sim_all)
        s['embed_top1_quantiles'] = {q: float(np.quantile(top1[top1 > 0], q))
                                     for q in (0.5, 0.9, 0.99)} if (top1 > 0).any() else {}

    pairs = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()
    pairs_path = os.path.join(args.out, 'overlap_pairs.csv')
    pairs.to_csv(pairs_path, index=False)
    exclude = sorted(set(pairs['id'])) if len(pairs) else []
    with open(os.path.join(args.out, 'exclude_ids.txt'), 'w') as fh:
        fh.write('\n'.join(exclude) + ('\n' if exclude else ''))
    with open(os.path.join(args.out, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f'Summary: {json.dumps(summary, indent=2)}')
    logger.info(f'{len(pairs)} flagged pairs -> {pairs_path}; {len(exclude)} training ids '
                f'to exclude -> {args.out}/exclude_ids.txt.')


if __name__ == '__main__':
    main()
