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
import hashlib
import logging
import argparse
import numpy as np
import pandas as pd
import torch

from energy.model import EnergyModel
from energy.losses import exact_nll, smoothed_nll, routing_regularizers, info_nce
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
    argp.add_argument('--smooth-tau', type=float, default=None,
                      help='Ablation A3: haversine label smoothing (their '
                           'tau=65) on top of the field. Tests whether the '
                           'hand-designed prior is subsumed by phi.')
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
    argp.add_argument('--resume', default=None,
                      help='Resume a run from its {run_name}_last.pt: restores model, '
                           'optimizer, LR scheduler, epoch, step, RNG and history, and '
                           'skips completed epochs. Missing file = start fresh (so a '
                           'requeued preempted job can always pass this). Mutually '
                           'exclusive with --init-from.')
    argp.add_argument('--eval-samples', type=int, default=20000,
                      help='Val rows used for the per-epoch metric pass.')
    argp.add_argument('--wandb', action='store_true', default=False,
                      help='Log training curves to Weights & Biases.')
    argp.add_argument('--wandb-project', default='spherical-pigeon')
    argp.add_argument('--wandb-entity', default=None)
    argp.add_argument('--wandb-mode', default='online', choices=['online', 'offline', 'disabled'],
                      help="Use 'offline' on compute nodes with no outbound internet "
                           "(e.g. Delta GPU nodes); sync afterwards with `wandb sync <dir>`.")
    argp.add_argument('--wandb-log-every', type=int, default=50,
                      help='Log a step-level training-loss point every N optimizer steps '
                           '(epochs over 5M rows are hours long — per-epoch-only logging '
                           'would leave the live graph flat for most of a run).')
    args = argp.parse_args()

    device = pick_device()
    logger.info(f'Device: {device}.')

    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError:
            raise SystemExit('--wandb given but the wandb package is not installed '
                              '(pip install wandb).')
        os.environ['WANDB_MODE'] = args.wandb_mode
        # Deterministic id from run_name so a requeued (preempted) job reattaches
        # to the same W&B run instead of spawning a duplicate.
        wandb_id = hashlib.md5(args.run_name.encode()).hexdigest()[:16]
        wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                               name=args.run_name, id=wandb_id, resume='allow',
                               config=vars(args))
        if args.wandb_mode == 'offline':
            logger.info(f'W&B in offline mode, writing to {wandb_run.dir}. '
                        f'Sync later with: wandb sync {os.path.dirname(wandb_run.dir)}')

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
    if args.init_from and args.resume:
        raise SystemExit('--init-from and --resume are mutually exclusive.')
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
    best_metric = math.inf  # lowest val median_km so far; {run_name}.pt tracks it
    best_epoch = -1
    start_epoch = 0

    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck['model'])
        if ck.get('optimizer') is not None:
            optimizer.load_state_dict(ck['optimizer'])
        if ck.get('scheduler') is not None:
            scheduler.load_state_dict(ck['scheduler'])
        global_step = ck.get('global_step', 0)
        history = ck.get('history', [])
        if ck.get('best_median_km') is not None:
            best_metric = ck['best_median_km']
        best_epoch = ck.get('best_epoch', -1)
        if ck.get('rng') is not None:
            rng.bit_generator.state = ck['rng']
        if ck.get('args', {}).get('epochs') not in (None, args.epochs):
            logger.warning(f"--resume checkpoint was for epochs={ck['args']['epochs']} "
                           f"but this run has epochs={args.epochs}; the LR schedule "
                           f"(CosineAnnealingLR T_max) will not line up.")
        start_epoch = ck.get('epoch', -1) + 1
        logger.info(f'Resumed from {args.resume}: continuing at epoch {start_epoch}/'
                    f'{args.epochs} (global_step {global_step}, best epoch {best_epoch}, '
                    f'best median_km {best_metric}).')
    elif args.resume:
        logger.info(f'--resume {args.resume} not found; starting a fresh run.')

    for epoch in range(start_epoch, args.epochs):
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
            elif args.smooth_tau is not None:
                loss = smoothed_nll(model, f, y_latlng, grid_latlngs, grid_rff,
                                    tau=args.smooth_tau,
                                    chunk_size=args.chunk_size,
                                    checkpoint_chunks=args.checkpoint_chunks).mean()
            else:
                nll, _, _ = exact_nll(model, f, y_idx, grid_rff,
                                      chunk_size=args.chunk_size,
                                      checkpoint_chunks=args.checkpoint_chunks)
                loss = nll.mean()

                if args.masks > 1:
                    warm = min(1.0, global_step / max(args.warmup_steps, 1))
                    loc_emb_t = model.location_tower.forward_features(grid_rff[y_idx])
                    r = model.routing_posterior(f, loc_emb_t)
                    regs = routing_regularizers(r, model.image_tower.gates())
                    loss = loss + warm * args.mu_conf * regs['confidence'] \
                                + warm * args.mu_bal * regs['balance'] \
                                + args.mu_l1 * regs['l1']

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * len(rows)
            epoch_n += len(rows)
            global_step += 1

            if wandb_run is not None and global_step % args.wandb_log_every == 0:
                wandb.log({'train/loss_step': loss.item(),
                           'train/grad_norm': float(grad_norm),
                           'train/lr': scheduler.get_last_lr()[0],
                           'epoch': epoch}, step=global_step)

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

        if wandb_run is not None:
            wandb.log({f'epoch/{k}': v for k, v in record.items() if k != 'epoch'},
                      step=global_step)

        ckpt = {'model': model.state_dict(), 'args': vars(args), 'epoch': epoch}

        # {run_name}.pt is the best epoch by val median_km (lower is better), not
        # the last one — a diverging run (0a' contrastive collapses after ep 1)
        # would otherwise leave the worst checkpoint behind for chaining/eval.
        # Do this first so {run_name}_last.pt below records the updated best_*.
        cur = record.get('median_km')
        if cur is None or cur < best_metric:
            if cur is not None:
                best_metric = cur
            best_epoch = epoch
            best_path = os.path.join(args.out, f'{args.run_name}.pt')
            torch.save({**ckpt, 'best_epoch': best_epoch, 'best_median_km': best_metric},
                       best_path + '.tmp')
            os.replace(best_path + '.tmp', best_path)
            logger.info(f'New best checkpoint: epoch {epoch}, median_km {cur}.')

        # {run_name}_last.pt carries the full training state so a preempted job
        # requeued onto a *-preempt partition resumes with --resume instead of
        # restarting from epoch 0. Written via a temp file + os.replace so a
        # preemption mid-write can't leave a truncated checkpoint.
        full = {**ckpt, 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'global_step': global_step,
                'history': history, 'best_epoch': best_epoch,
                'best_median_km': best_metric, 'rng': rng.bit_generator.state}
        last_path = os.path.join(args.out, f'{args.run_name}_last.pt')
        torch.save(full, last_path + '.tmp')
        os.replace(last_path + '.tmp', last_path)

        hist_path = os.path.join(args.out, f'{args.run_name}_history.json')
        with open(hist_path + '.tmp', 'w') as fh:
            json.dump(history, fh, indent=2)
        os.replace(hist_path + '.tmp', hist_path)

    if start_epoch >= args.epochs:
        logger.info(f'--resume checkpoint is already at epoch {start_epoch - 1}/'
                    f'{args.epochs}; nothing to do.')
    logger.info(f'Training complete. Best epoch {best_epoch} '
                f'(median_km {best_metric:.3f}) -> {args.run_name}.pt; '
                f'last epoch -> {args.run_name}_last.pt.')
    if wandb_run is not None:
        wandb.finish()


if __name__ == '__main__':
    main()
