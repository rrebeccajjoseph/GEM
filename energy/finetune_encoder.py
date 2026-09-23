"""Stage C — unfreeze the last N CLIP blocks and finetune end to end.

Cached embeddings no longer suffice here: gradients must reach the encoder,
so training runs from images. The towers are initialized from a Stage A/B
checkpoint; the encoder gets lr 1e-5 with optional layer-wise decay 0.8, the
rest 1e-4 (matching the plan, which mirrors PIGEON's "unfreeze the last CLIP
layer(s)" so the comparison stays fair).

The full-grid location forward is negligible next to the ViT-L forward at
this stage — no chunking heroics needed beyond the usual streaming logsumexp.

After training, REBUILD the embedding cache with --encoder-out before
running Stage D / E on top: the cached embeddings are stale the moment the
encoder moves.

Usage:
    PIGEON_CLIP_MODEL=geolocal/StreetCLIP python -m energy.finetune_encoder \
        --init-from saved_models/energy/stage_b.pt \
        [--unfreeze-blocks 2] [--llrd 0.8] [--epochs 5]
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
from torch import nn

from energy.model import EnergyModel
from energy.losses import exact_nll
from energy.grid import build_grid, cell_index_map, snap_to_grid

logger = logging.getLogger('energy.finetune_encoder')
logging.basicConfig(level=logging.INFO)


def encoder_param_groups(embedding_module: nn.Module, blocks: list,
                         final_module: nn.Module, n_unfreeze: int,
                         base_lr: float, llrd: float=1.0):
    """Freezes all but the last n_unfreeze blocks; builds LLRD param groups.

    Pure module-list logic (testable without transformers): the deepest
    unfrozen block gets base_lr, each earlier one base_lr * llrd^k, the
    embedding module stays frozen, the final norm trains at base_lr.

    Args:
        embedding_module (nn.Module): patch/pos embedding (always frozen)
        blocks (list): transformer blocks, input-to-output order
        final_module (nn.Module): final layernorm (trains at base_lr)
        n_unfreeze (int): number of trailing blocks to unfreeze
        base_lr (float): lr of the deepest unfrozen block
        llrd (float, optional): layer-wise lr decay factor.

    Returns:
        list: optimizer param groups for the unfrozen encoder parameters
    """
    for p in embedding_module.parameters():
        p.requires_grad_(False)
    for block in blocks[:len(blocks) - n_unfreeze]:
        for p in block.parameters():
            p.requires_grad_(False)

    groups = []
    unfrozen = blocks[len(blocks) - n_unfreeze:]
    for depth_from_top, block in enumerate(reversed(unfrozen)):
        lr = base_lr * (llrd ** depth_from_top)
        groups.append({'params': list(block.parameters()), 'lr': lr})
    groups.append({'params': list(final_module.parameters()), 'lr': base_lr})
    return groups


class CLIPEncoder(nn.Module):
    """Mean-pooled CLIPVisionModel, matching embed_cache / CLIPEmbedding."""

    def __init__(self, model_name: str):
        from transformers import CLIPVisionModel
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained(model_name)

    @property
    def parts(self):
        # transformers >=4.5x flattened CLIPVisionModel: embeddings/encoder/
        # post_layernorm live directly on it, not nested under .vision_model
        # (which no longer exists). Support both so this doesn't silently
        # break again on a transformers downgrade/upgrade either direction.
        base = getattr(self.clip, 'vision_model', self.clip)
        return base.embeddings, list(base.encoder.layers), base.post_layernorm

    def forward(self, pixel_values):
        out = self.clip.base_model(pixel_values=pixel_values)
        return out.last_hidden_state.mean(dim=1)


def train_stage_c(encoder: nn.Module, model: EnergyModel, loader,
                  grid_rff: torch.Tensor, encoder_groups: list,
                  epochs: int=5, tower_lr: float=1e-4,
                  chunk_size: int=32768, device: str='cpu',
                  log_every: int=100, on_epoch_end=None, wandb_run=None,
                  resume_state: dict=None, ckpt_every: int=2000, on_ckpt=None):
    """The Stage C loop, generic over the encoder (testable with a stub).

    Args:
        encoder: module mapping pixel batches -> [B, in_dim] embeddings
        model: EnergyModel with towers initialized from Stage A/B
        loader: yields (pixels, cell_idx) batches
        grid_rff: fixed grid RFF features [G, F]
        encoder_groups: param groups from encoder_param_groups
        on_epoch_end (callable, optional): callback(epoch, mean_loss, optimizer,
            global_step) — full-epoch summary.
        on_ckpt (callable, optional): callback(epoch, optimizer, global_step)
            every `ckpt_every` steps — mid-epoch insurance. A full pass at
            this dataset's scale runs many hours; on_epoch_end alone crashed
            a ~21h combined-corpus run to zero checkpoints (a truncated
            image several hours in, 2026-09-08). This does NOT resume the
            exact batch position — the DataLoader isn't seeded per-epoch —
            it just restarts the current epoch's pass with the checkpointed
            weights/optimizer rather than from scratch, which is the
            difference that matters at this scale.
        resume_state (dict, optional): {'optimizer', 'start_epoch',
            'global_step'} — start_epoch is the epoch to *redo* the pass
            for (not epoch + 1), since a mid-epoch checkpoint has no
            reliable resume position within it.
    """
    encoder = encoder.to(device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        encoder_groups + [{'params': model.parameters(), 'lr': tower_lr}],
        weight_decay=0.01)

    global_step = 0
    start_epoch = 0
    if resume_state is not None:
        optimizer.load_state_dict(resume_state['optimizer'])
        start_epoch = resume_state['start_epoch']
        global_step = resume_state['global_step']
        logger.info(f'Resuming Stage C training: redoing epoch {start_epoch}\'s '
                    f'pass from the checkpointed weights (global_step {global_step}).')

    for epoch in range(start_epoch, epochs):
        encoder.train()
        model.train()
        total, n = 0.0, 0
        for step, (pixels, y_idx) in enumerate(loader):
            pixels, y_idx = pixels.to(device), y_idx.to(device)
            f = encoder(pixels)
            nll, _, _ = exact_nll(model, f, y_idx, grid_rff,
                                  chunk_size=chunk_size)
            loss = nll.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g['params']], 1.0)
            optimizer.step()

            total += loss.item() * len(y_idx)
            n += len(y_idx)
            global_step += 1
            if step % log_every == 0:
                logger.info(f'epoch {epoch} step {step}: nll {loss.item():.3f}')
                if wandb_run is not None:
                    import wandb
                    wandb.log({'train/loss_step': loss.item(),
                               'train/grad_norm': float(grad_norm),
                               'epoch': epoch}, step=global_step)
            if on_ckpt is not None and ckpt_every and global_step % ckpt_every == 0:
                on_ckpt(epoch, optimizer, global_step)

        if on_epoch_end is not None:
            on_epoch_end(epoch, total / max(n, 1), optimizer, global_step)
        if wandb_run is not None:
            import wandb
            wandb.log({'epoch/train_loss': total / max(n, 1), 'epoch': epoch},
                      step=global_step)

    return encoder, model


class TrainImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths, cell_idx, processor):
        self.paths = paths
        self.cell_idx = cell_idx
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image, ImageFile
        # A truncated JPEG (common at MP-16-Pro scale — some source URLs were
        # already partially served) otherwise raises OSError and crashes the
        # whole run; energy/embed_cache.py already tolerates this for
        # benchmark images the same way. Any *other* unreadable file (zero
        # bytes, wrong format entirely) falls back to a different random row
        # rather than losing a run that can be many hours into its only
        # epoch — one bad row silently skipped is a far smaller cost than
        # a full restart (confirmed the hard way: a truncated MP-16 image
        # crashed a ~21h combined-corpus run with zero checkpoint to resume
        # from, 2026-09-08).
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            image = Image.open(self.paths[i]).convert('RGB')
        except Exception as e:
            logger.warning(f'Unreadable image at row {i} ({self.paths[i]}): {e}; '
                           f'substituting a random other row.')
            j = np.random.randint(len(self.paths))
            return self.__getitem__(j)
        pixels = self.processor(images=image, return_tensors='pt')['pixel_values']
        return pixels.squeeze(0), self.cell_idx[i]


def main():
    from transformers import CLIPProcessor
    from config import CLIP_MODEL, METADATA_PATH_OSV, IMAGE_PATH_OSV
    from energy.train import pick_device, load_grid

    argp = argparse.ArgumentParser(description='Stage C encoder finetuning.')
    argp.add_argument('--init-from', default=None,
                      help='Stage A/B checkpoint for the towers. Required unless '
                           '--resume is given (a resumed run recovers the tower '
                           'config from its own last checkpoint).')
    argp.add_argument('--grid', default='data/energy/grid.npz')
    argp.add_argument('--metadata', default=METADATA_PATH_OSV)
    argp.add_argument('--images', default=IMAGE_PATH_OSV)
    argp.add_argument('--out', default='saved_models/energy')
    argp.add_argument('--run-name', default='stage_c')
    argp.add_argument('--unfreeze-blocks', type=int, default=2)
    argp.add_argument('--encoder-lr', type=float, default=1e-5)
    argp.add_argument('--llrd', type=float, default=1.0,
                      help='Layer-wise lr decay (0.8 if unstable).')
    argp.add_argument('--tower-lr', type=float, default=1e-4)
    argp.add_argument('--epochs', type=int, default=5)
    argp.add_argument('--batch-size', type=int, default=64)
    argp.add_argument('--num-workers', type=int, default=8)
    argp.add_argument('--max-rows', type=int, default=None,
                      help='Subsample the training set to this many rows '
                           '(random, seed 330) — a full epoch over all of '
                           'OSV-5M at real ViT-L gradient cost can take days; '
                           'this bounds Stage C to a single job\'s walltime.')
    argp.add_argument('--encoder-out', default=None,
                      help='Directory to save the finetuned encoder + processor '
                           '(HF save_pretrained), rewritten after every epoch so '
                           'a killed job still leaves a usable encoder. Point '
                           'PIGEON_CLIP_MODEL at this directory to rebuild the '
                           'embedding cache with the finetuned weights.')
    argp.add_argument('--resume', default=None,
                      help='Resume from {run_name}_last.pt: restores model, '
                           'encoder, optimizer and epoch, skipping completed '
                           'epochs. A full pass over a multi-million-row '
                           'combined dataset is long enough that losing it to '
                           'a preemption or a stray kill is expensive. Missing '
                           'file = start fresh. Mutually exclusive with '
                           '--init-from.')
    argp.add_argument('--ckpt-every', type=int, default=2000,
                      help='Write {run_name}_last.pt every N optimizer steps, '
                           'not just at epoch end — a single epoch at multi-'
                           'million-row scale can run most of a day, and '
                           'epoch-only checkpointing lost an entire ~21h run '
                           'to one bad image with nothing to resume from '
                           '(2026-09-08). A mid-epoch checkpoint restarts the '
                           'current epoch\'s pass rather than resuming a '
                           'specific batch position (the loader isn\'t seeded '
                           'per-epoch), which is still far cheaper than redoing '
                           'the whole run.')
    argp.add_argument('--wandb', action='store_true', default=False,
                      help='Log training curves to Weights & Biases.')
    argp.add_argument('--wandb-project', default='spherical-pigeon')
    argp.add_argument('--wandb-entity', default=None)
    argp.add_argument('--wandb-mode', default='online', choices=['online', 'offline', 'disabled'])
    args = argp.parse_args()

    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError:
            raise SystemExit('--wandb given but the wandb package is not installed '
                              '(pip install wandb).')
        os.environ['WANDB_MODE'] = args.wandb_mode
        wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                               name=args.run_name, config=vars(args))

    if args.init_from and args.resume:
        raise SystemExit('--init-from and --resume are mutually exclusive.')
    if not args.init_from and not args.resume:
        raise SystemExit('--init-from is required unless --resume points to an '
                          'existing checkpoint.')

    device = pick_device()
    latlngs_np, resolution, _ = load_grid(args.grid, want_rasters=False)

    resume_ckpt = None
    if args.resume and os.path.exists(args.resume):
        resume_ckpt = torch.load(args.resume, map_location=device)
        coarse_args = resume_ckpt['args']
        state = resume_ckpt
        logger.info(f'Resuming from {args.resume}.')
    elif args.resume:
        logger.info(f'--resume {args.resume} not found; --init-from required '
                     'to start fresh.')
        if not args.init_from:
            raise SystemExit('--resume checkpoint missing and no --init-from given.')
        state = torch.load(args.init_from, map_location=device)
        coarse_args = state['args']
    else:
        state = torch.load(args.init_from, map_location=device)
        coarse_args = state['args']

    raster_table = None
    if coarse_args.get('rasters'):
        _, _, raster_table = load_grid(args.grid, want_rasters=True)
    model = EnergyModel(in_dim=1024, d=coarse_args['d'],
                        n_masks=coarse_args['masks'], raster_table=raster_table,
                        use_season=coarse_args.get('season', False),
                        gated=coarse_args.get('gate', False))
    model.load_state_dict(state['model'])
    model = model.to(device)  # must happen before encode_features below —
    # train_stage_c() also moves it, but only after that call already runs.

    encoder = CLIPEncoder(CLIP_MODEL)
    if resume_ckpt is not None:
        encoder.load_state_dict(resume_ckpt['encoder'])
    embeddings_mod, blocks, final_norm = encoder.parts
    groups = encoder_param_groups(embeddings_mod, blocks, final_norm,
                                  args.unfreeze_blocks, args.encoder_lr,
                                  args.llrd)
    n_trainable = sum(p.numel() for g in groups for p in g['params'])
    logger.info(f'Encoder: {args.unfreeze_blocks} blocks unfrozen '
                f'({n_trainable / 1e6:.1f}M params).')

    meta = pd.read_csv(args.metadata, dtype={'id': str})
    meta = meta[meta['selection'] == 'train']
    if args.max_rows and args.max_rows < len(meta):
        meta = meta.sample(n=args.max_rows, random_state=330)
        logger.info(f'Subsampled training set to {len(meta)} rows '
                    f'(--max-rows {args.max_rows}).')
    cells, _ = build_grid(resolution)
    cell_idx = snap_to_grid(meta['lat'].values, meta['lng'].values,
                            resolution, cell_index_map(cells))
    paths = [os.path.join(args.images, p) for p in meta['image'].values]

    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    loader = torch.utils.data.DataLoader(
        TrainImageDataset(paths, cell_idx, processor),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        drop_last=True)

    grid_rff = model.location_tower.encode_features(
        torch.from_numpy(latlngs_np).float().to(device))

    os.makedirs(args.out, exist_ok=True)

    def save(epoch, mean_loss, optimizer, global_step):
        logger.info(f'epoch {epoch}: mean nll {mean_loss:.3f}')
        ckpt = {'model': model.state_dict(), 'encoder': encoder.state_dict(),
                'args': {**coarse_args, **vars(args)}, 'epoch': epoch,
                'optimizer': optimizer.state_dict(), 'start_epoch': epoch + 1,
                'global_step': global_step}
        # {run_name}.pt is the latest-epoch snapshot (Stage C has no val split
        # to pick a "best" epoch by); {run_name}_last.pt is the same content
        # under the name --resume looks for. Both written via a temp file so a
        # kill mid-write can't leave a truncated checkpoint for either.
        for name in (f'{args.run_name}.pt', f'{args.run_name}_last.pt'):
            path = os.path.join(args.out, name)
            torch.save(ckpt, path + '.tmp')
            os.replace(path + '.tmp', path)
        # Rewritten every epoch, not just at the end — a run over millions of
        # rows can take days, and this is the artifact embed_cache reads next;
        # losing it to a preemption after epoch 1 of 2 would be expensive.
        if args.encoder_out:
            os.makedirs(args.encoder_out, exist_ok=True)
            encoder.clip.save_pretrained(args.encoder_out)
            processor.save_pretrained(args.encoder_out)
            logger.info(f'Saved encoder + processor to {args.encoder_out} '
                        f'(epoch {epoch}).')

    def save_ckpt(epoch, optimizer, global_step):
        # Mid-epoch insurance: same content as `save`'s end-of-epoch write,
        # but start_epoch=epoch (redo this epoch's pass), not epoch+1 — there
        # is no reliable resume position *within* an epoch since the loader
        # isn't seeded, so a mid-epoch checkpoint means "restart this epoch
        # from these weights," not "continue partway through it."
        logger.info(f'mid-epoch checkpoint: epoch {epoch}, global_step {global_step}.')
        ckpt = {'model': model.state_dict(), 'encoder': encoder.state_dict(),
                'args': {**coarse_args, **vars(args)}, 'epoch': epoch,
                'optimizer': optimizer.state_dict(), 'start_epoch': epoch,
                'global_step': global_step}
        path = os.path.join(args.out, f'{args.run_name}_last.pt')
        torch.save(ckpt, path + '.tmp')
        os.replace(path + '.tmp', path)
        if args.encoder_out:
            os.makedirs(args.encoder_out, exist_ok=True)
            encoder.clip.save_pretrained(args.encoder_out)
            processor.save_pretrained(args.encoder_out)

    resume_state = None
    if resume_ckpt is not None:
        resume_state = {'optimizer': resume_ckpt['optimizer'],
                        'start_epoch': resume_ckpt['start_epoch'],
                        'global_step': resume_ckpt['global_step']}
        if resume_state['start_epoch'] >= args.epochs:
            logger.info(f"--resume checkpoint is already at epoch "
                        f"{resume_state['start_epoch'] - 1}/{args.epochs}; "
                        f"nothing to do.")

    train_stage_c(encoder, model, loader, grid_rff, groups,
                  epochs=args.epochs, tower_lr=args.tower_lr, device=device,
                  on_epoch_end=save, wandb_run=wandb_run,
                  resume_state=resume_state, ckpt_every=args.ckpt_every,
                  on_ckpt=save_ckpt)
    logger.info('Stage C complete. Rebuild the embedding cache with this '
                'encoder before Stage D/E.')

    if wandb_run is not None:
        wandb.finish()


if __name__ == '__main__':
    main()
