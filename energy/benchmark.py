"""Benchmark evaluation: encode Im2GPS3k / YFCC4k / YFCC26k / OSV-5M-test
through the SAME encoder as training and run the full metric suite —
coarse posterior metrics (distance thresholds + calibration) and, given a
Stage E checkpoint, refined predictions.

Reads the repo's data/benchmarks/benchmarks.json (name -> meta CSV + image
dir) so both pipelines evaluate on identical files. Benchmark CSVs vary in
column naming; lat/lng and image columns are auto-detected.

Benchmark embeddings are cached under --embed-cache (a few thousand images
each — cheap to redo if the encoder changes, but versioned by encoder name).

Usage:
    PIGEON_CLIP_MODEL=geolocal/StreetCLIP python -m energy.benchmark \
        --coarse saved_models/energy/stage_b.pt --benchmark im2gps3k \
        [--refiner saved_models/energy/stage_e.pt] [--grid data/energy/grid.npz]
"""

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)

import json
import logging
import argparse
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger('energy.benchmark')
logging.basicConfig(level=logging.INFO)

LAT_COLUMNS = ['lat', 'latitude', 'LAT', 'Latitude']
LNG_COLUMNS = ['lng', 'longitude', 'lon', 'LON', 'Longitude']
IMG_COLUMNS = ['image', 'img', 'img_path', 'image_path', 'IMG_ID', 'id', 'photo_id']


def detect_columns(df: pd.DataFrame):
    """Finds the lat / lng / image columns of a benchmark CSV.

    Returns:
        tuple: (lat_col, lng_col, img_col)
    """
    def pick(candidates, what):
        for c in candidates:
            if c in df.columns:
                return c
        raise ValueError(f'Could not detect {what} column in {list(df.columns)}; '
                         f'expected one of {candidates}.')

    return (pick(LAT_COLUMNS, 'latitude'), pick(LNG_COLUMNS, 'longitude'),
            pick(IMG_COLUMNS, 'image'))


def load_benchmark(name: str):
    """Loads a benchmark's metadata and resolves image paths.

    Returns:
        tuple: (image_paths list, latlngs np.ndarray [N, 2])
    """
    from config import BENCHMARKS

    with open(BENCHMARKS) as fh:
        benchmarks = json.load(fh)
    assert name in benchmarks, \
        f'{name} not in {sorted(benchmarks)} (see {BENCHMARKS}).'

    spec = benchmarks[name]
    df = pd.read_csv(spec['meta'])
    lat_col, lng_col, img_col = detect_columns(df)

    paths = []
    for v in df[img_col].astype(str).values:
        p = v if os.path.isabs(v) else os.path.join(spec['images'], v)
        if not os.path.splitext(p)[1]:
            p += '.jpg'
        paths.append(p)

    missing = [p for p in paths[:100] if not os.path.exists(p)]
    if missing:
        logger.warning(f'{name}: {len(missing)}/100 sampled image paths missing, '
                       f'e.g. {missing[0]} — check the images dir in benchmarks.json.')

    latlngs = df[[lat_col, lng_col]].values.astype(np.float32)
    return paths, latlngs


def encode_benchmark(name: str, paths, embed_cache_dir: str,
                     batch_size: int=128, device: str='cpu') -> np.ndarray:
    """Encodes benchmark images with the configured encoder, cached on disk
    keyed by (benchmark, encoder)."""
    from config import CLIP_MODEL, CLIP_EMBED_DIM
    from transformers import CLIPProcessor, CLIPVisionModel
    from energy.embed_cache import ImageDataset

    encoder_tag = CLIP_MODEL.replace('/', '_')
    cache_path = os.path.join(embed_cache_dir, f'{name}.{encoder_tag}.npy')
    if os.path.exists(cache_path):
        emb = np.load(cache_path)
        if emb.shape[0] == len(paths):
            logger.info(f'Loaded cached embeddings: {cache_path}.')
            return emb
        logger.warning(f'Cached embedding count mismatch, re-encoding {name}.')

    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model = CLIPVisionModel.from_pretrained(CLIP_MODEL).to(device).eval()
    loader = torch.utils.data.DataLoader(ImageDataset(paths, processor),
                                         batch_size=batch_size, num_workers=4)
    out = np.zeros((len(paths), CLIP_EMBED_DIM), dtype=np.float32)
    row = 0
    with torch.no_grad():
        for pixels in loader:
            e = model.base_model(pixel_values=pixels.to(device)) \
                     .last_hidden_state.mean(dim=1)
            out[row:row + e.shape[0]] = e.cpu().numpy()
            row += e.shape[0]

    os.makedirs(embed_cache_dir, exist_ok=True)
    np.save(cache_path, out)
    logger.info(f'Encoded {row} images -> {cache_path}.')
    return out


def evaluate_benchmark(coarse, embeddings: np.ndarray, latlngs: np.ndarray,
                       grid_rff: torch.Tensor, grid_latlngs: np.ndarray,
                       refiner=None, bank=None, topk: int=20,
                       device: str='cpu') -> dict:
    """Full metric suite: coarse posterior metrics, plus refined-prediction
    distance metrics when a Stage E scorer and CandidateBank are given.
    Pure tensor path — no image or transformers dependency (testable)."""
    from energy.evaluation import evaluate_on_cache, distance_metrics
    from energy.refine import mine_topk, refine_predictions

    emb_t = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))
    metrics = {f'coarse_{k}': v for k, v in
               evaluate_on_cache(coarse, emb_t, latlngs, grid_rff, grid_latlngs,
                                 device=device).items()}

    if refiner is not None:
        assert bank is not None, 'Refined evaluation needs a CandidateBank.'
        topk_idx, topk_logp = mine_topk(coarse, embeddings, grid_rff, k=topk,
                                        device=device)
        out = refine_predictions(refiner, embeddings, topk_idx, topk_logp,
                                 bank, device=device)
        refined = distance_metrics(out['pred_latlng'], latlngs)
        metrics.update({f'refined_{k}': v for k, v in refined.items()})
        metrics['refined_mean_topk_coverage'] = float(out['coverage'].mean())

    return metrics


def main():
    from energy.train import pick_device, load_grid
    from energy.model import EnergyModel
    from energy.refine import JointMLPScorer, CandidateBank
    from energy.grid import build_grid

    argp = argparse.ArgumentParser(description='Benchmark evaluation.')
    argp.add_argument('--coarse', required=True)
    argp.add_argument('--refiner', default=None)
    argp.add_argument('--benchmark', required=True,
                      help='Name from data/benchmarks/benchmarks.json.')
    argp.add_argument('--grid', default='data/energy/grid.npz')
    argp.add_argument('--embed-cache', default='data/energy/benchmark_cache')
    argp.add_argument('--topk', type=int, default=20)
    argp.add_argument('--out', default='saved_models/energy')
    args = argp.parse_args()

    device = pick_device()
    latlngs_np, resolution, _ = load_grid(args.grid, want_rasters=False)

    state = torch.load(args.coarse, map_location=device)
    coarse_args = state['args']
    raster_table = None
    if coarse_args.get('rasters'):
        _, _, raster_table = load_grid(args.grid, want_rasters=True)
    coarse = EnergyModel(in_dim=1024, d=coarse_args['d'],
                         n_masks=coarse_args['masks'], raster_table=raster_table,
                         use_season=coarse_args.get('season', False)).to(device)
    coarse.load_state_dict(state['model'])
    coarse.eval()

    grid_rff = coarse.location_tower.encode_features(
        torch.from_numpy(latlngs_np).float().to(device))

    refiner, bank = None, None
    if args.refiner:
        rstate = torch.load(args.refiner, map_location=device)
        refiner = JointMLPScorer(in_dim=1024)
        refiner.load_state_dict(rstate['scorer'])
        cells, _ = build_grid(resolution)
        bank = CandidateBank(cells, resolution, rstate['args']['fine_res'])

    paths, bench_latlngs = load_benchmark(args.benchmark)
    embeddings = encode_benchmark(args.benchmark, paths, args.embed_cache,
                                  device=device)

    metrics = evaluate_benchmark(coarse, embeddings, bench_latlngs, grid_rff,
                                 latlngs_np, refiner=refiner, bank=bank,
                                 topk=args.topk, device=device)
    metrics = {k: v for k, v in metrics.items() if not isinstance(v, list)}
    logger.info(json.dumps(metrics, indent=2))

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f'benchmark_{args.benchmark}.json')
    with open(out_path, 'w') as fh:
        json.dump(metrics, fh, indent=2)
    logger.info(f'Saved to {out_path}.')


if __name__ == '__main__':
    main()
