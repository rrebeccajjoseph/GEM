"""Trainer for the coarse energy model over cached embeddings.

Stages map to flags (each stage changes one thing):
    A  (default)          M=1, no rasters, no season — exact NLL only
    B  --rasters          + raster compatibility terms
    D  --masks 16 --season  + latent masks and the season latent
    0a' --contrastive     same towers, InfoNCE instead of exact NLL

Stage C (encoder unfreezing) is not served by this trainer — it needs image
gradients, not cached embeddings.

Usage:
    python -m energy.train --cache data/energy/cache --grid data/energy/grid.npz \
        [--rasters] [--masks 16] [--season] [--contrastive] [--epochs 20]
"""

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)

import json
import math
import logging
import argparse
import numpy as np
import pandas as pd
import torch

from energy.model import EnergyModel
from energy.losses import exact_nll, routing_regularizers, info_nce
from energy.evaluation import evaluate_on_cache
from energy.grid import build_grid, cell_index_map, snap_to_grid

logger = logging.getLogger('energy.train')
logging.basicConfig(level=logging.INFO)


def pick_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def load_grid(grid_path: str, want_rasters: bool):
    """Loads the grid npz; returns (latlngs [G,2], resolution, raster_table|None)."""
    data = np.load(grid_path, allow_pickle=False)
    latlngs = data['latlngs']
    resolution = int(data['resolution'])
    table = None
    if want_rasters:
        table = {k[len('raster_'):]: data[k] for k in data.files
                 if k.startswith('raster_')}
        if not table:
            raise ValueError(f'--rasters given but {grid_path} has no raster_* arrays.')
    return latlngs, resolution, table


def load_cache(cache_dir: str):
    """Loads the embedding memmap and row-aligned index."""
    index = pd.read_csv(os.path.join(cache_dir, 'index.csv'), dtype={'id': str})
    embeddings = np.load(os.path.join(cache_dir, 'embeddings.f16.npy'), mmap_mode='r')
    assert len(index) == embeddings.shape[0], 'index/embeddings row mismatch'
    return embeddings, index


def main():
    argp = argparse.ArgumentParser(description='Train the coarse energy model.')
    argp.add_argument('--cache', default='data/energy/cache')
    argp.add_argument('--grid', default='data/energy/grid.npz')
    argp.add_argument('--out', default='saved_models/energy')
    argp.add_argument('--run-name', default='stage_a')
    argp.add_argument('--epochs', type=int, default=20)
    argp.add_argument('--batch-size', type=int, default=256)
    argp.add_argument('--lr', type=float, default=3e-4)
    argp.add_argument('--head-lr', type=float, default=1e-3,
                      help='LR for raster/season heads (Stage B: new heads).')
    argp.add_argument('--weight-decay', type=float, default=0.01)
    argp.add_argument('--d', type=int, default=512)
    argp.add_argument('--masks', type=int, default=1)
    argp.add_argument('--rasters', action='store_true', default=False)
    argp.add_argument('--season', action='store_true', default=False)
    argp.add_argument('--contrastive', action='store_true', default=False,
                      help="Ablation 0a': InfoNCE with in-batch negatives.")
    argp.add_argument('--chunk-size', type=int, default=32768)
    argp.add_argument('--checkpoint-chunks', action='store_true', default=False,
                      help='Gradient-checkpoint grid chunks (use when masks > 1).')
    argp.add_argument('--mu-conf', type=float, default=0.01)
    argp.add_argument('--mu-bal', type=float, default=0.1)
    argp.add_argument('--mu-l1', type=float, default=1e-4)
    argp.add_argument('--warmup-steps', type=int, default=2000,
                      help='Anneal mu_conf and mu_bal from 0 over this many steps.')
    argp.add_argument('--init-from', default=None,
                      help='Checkpoint to initialize from (stage chaining).')
    argp.add_argument('--eval-samples', type=int, default=20000,
                      help='Val rows used for the per-epoch metric pass.')
    args = argp.parse_args()

    device = pick_device()
    logger.info(f'Device: {device}.')

    # Grid
    latlngs_np, resolution, raster_table = load_grid(args.grid, args.rasters)
    grid_latlngs = torch.from_numpy(latlngs_np).float().to(device)
    G = grid_latlngs.shape[0]
    logger.info(f'Grid: {G} cells at H3 res {resolution}.')

    # Data
    embeddings, index = load_cache(args.cache)
    cells, _ = build_grid(resolution)
    idx_map = cell_index_map(cells)
    logger.info('Snapping targets to grid cells.')
    index['cell_idx'] = snap_to_grid(index['lat'].values, index['lng'].values,
                                     resolution, idx_map)

    splits = {name: np.flatnonzero((index['selection'] == name).values)
              for name in ['train', 'val']}
    logger.info(f"Rows: train {len(splits['train'])}, val {len(splits['val'])}.")

    # Model
    model = EnergyModel(in_dim=embeddings.shape[1], d=args.d, n_masks=args.masks,
                        raster_table=raster_table, use_season=args.season).to(device)
    if args.init_from:
        state = torch.load(args.init_from, map_location=device)
        missing, unexpected = model.load_state_dict(state['model'], strict=False)
        logger.info(f'Initialized from {args.init_from} '
                    f'(missing: {len(missing)}, unexpected: {len(unexpected)}).')

    # Fixed RFF encoding of the grid — precomputed once (centroids never move);
    # the location-tower MLP on top still runs in-graph every step.
    grid_rff = model.location_tower.encode_features(grid_latlngs)

    tower_params, head_params = [], []
    for name, p in model.named_parameters():
        (head_params if name.startswith(('rasters.', 'season.')) else tower_params).append(p)
    optimizer = torch.optim.AdamW(
        [{'params': tower_params, 'lr': args.lr},
         {'params': head_params, 'lr': args.head_lr}],
        weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(splits['train']) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * steps_per_epoch)

    os.makedirs(args.out, exist_ok=True)
    train_latlng_np = index[['lat', 'lng']].values.astype(np.float32)
    train_cells_np = index['cell_idx'].values

    rng = np.random.default_rng(330)
    global_step = 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(splits['train'])
        epoch_loss, epoch_n = 0.0, 0

        for start in range(0, len(order), args.batch_size):
            rows = order[start:start + args.batch_size]
            f = torch.from_numpy(np.asarray(embeddings[rows], dtype=np.float32)).to(device)
            y_latlng = torch.from_numpy(train_latlng_np[rows]).to(device)
            y_idx = torch.from_numpy(train_cells_np[rows]).to(device)

            if args.contrastive:
                loss = info_nce(model, f, y_latlng)
            else:
                nll, _, _ = exact_nll(model, f, y_idx, grid_rff,
                                      target_latlng=y_latlng,
                                      chunk_size=args.chunk_size,
                                      checkpoint_chunks=args.checkpoint_chunks)
                loss = nll.mean()

                if args.masks > 1:
                    warm = min(1.0, global_step / max(args.warmup_steps, 1))
                    loc_emb_t = model.location_tower(y_latlng)
                    r = model.routing_posterior(f, loc_emb_t)
                    regs = routing_regularizers(r, model.image_tower.gates())
                    loss = loss + warm * args.mu_conf * regs['confidence'] \
                                + warm * args.mu_bal * regs['balance'] \
                                + args.mu_l1 * regs['l1']

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * len(rows)
            epoch_n += len(rows)
            global_step += 1

        train_loss = epoch_loss / epoch_n

        # Per-epoch diagnostics: the two things most likely to silently degenerate
        diag = {}
        if model.rasters is not None:
            for name, head in model.rasters.heads.items():
                if hasattr(head, 'log_lambda'):
                    diag[f'lambda_{name}'] = float(
                        torch.nn.functional.softplus(head.log_lambda))
        if args.masks > 1:
            gates = model.image_tower.gates()
            diag['gate_mean'] = float(gates.mean())
            diag['gate_active_frac'] = float((gates > 0.5).float().mean())
        # Tripwire for the negative-phase bug (norms inflating on popular cells)
        with torch.no_grad():
            h_norms = model.location_tower.forward_features(
                grid_rff[::max(1, G // 4096)]).norm(dim=-1)
            diag['h_norm_p50'] = float(h_norms.median())
            diag['h_norm_p99'] = float(h_norms.quantile(0.99))

        # Val metrics on a subsample
        val_rows = splits['val'][:args.eval_samples]
        val_emb = torch.from_numpy(np.asarray(embeddings[val_rows], dtype=np.float32))
        metrics = evaluate_on_cache(model, val_emb, train_latlng_np[val_rows],
                                    grid_rff, latlngs_np, device=device)
        model.train()

        record = {'epoch': epoch, 'train_loss': train_loss, **diag,
                  **{k: v for k, v in metrics.items() if not isinstance(v, list)}}
        history.append(record)
        logger.info(json.dumps(record))

        torch.save({'model': model.state_dict(), 'args': vars(args), 'epoch': epoch},
                   os.path.join(args.out, f'{args.run_name}.pt'))
        with open(os.path.join(args.out, f'{args.run_name}_history.json'), 'w') as fh:
            json.dump(history, fh, indent=2)

    logger.info('Training complete.')


if __name__ == '__main__':
    main()
