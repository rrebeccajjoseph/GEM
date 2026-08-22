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
        vm = self.clip.vision_model
        return vm.embeddings, list(vm.encoder.layers), vm.post_layernorm

    def forward(self, pixel_values):
        out = self.clip.base_model(pixel_values=pixel_values)
        return out.last_hidden_state.mean(dim=1)


def train_stage_c(encoder: nn.Module, model: EnergyModel, loader,
                  grid_rff: torch.Tensor, encoder_groups: list,
                  epochs: int=5, tower_lr: float=1e-4,
                  chunk_size: int=32768, device: str='cpu',
                  log_every: int=100, on_epoch_end=None):
    """The Stage C loop, generic over the encoder (testable with a stub).

    Args:
        encoder: module mapping pixel batches -> [B, in_dim] embeddings
        model: EnergyModel with towers initialized from Stage A/B
        loader: yields (pixels, cell_idx) batches
        grid_rff: fixed grid RFF features [G, F]
        encoder_groups: param groups from encoder_param_groups
        on_epoch_end (callable, optional): callback(epoch, mean_loss)
    """
    encoder = encoder.to(device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        encoder_groups + [{'params': model.parameters(), 'lr': tower_lr}],
        weight_decay=0.01)

    for epoch in range(epochs):
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
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g['params']], 1.0)
            optimizer.step()

            total += loss.item() * len(y_idx)
            n += len(y_idx)
            if step % log_every == 0:
                logger.info(f'epoch {epoch} step {step}: nll {loss.item():.3f}')

        if on_epoch_end is not None:
            on_epoch_end(epoch, total / max(n, 1))

    return encoder, model


class TrainImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths, cell_idx, processor):
        self.paths = paths
        self.cell_idx = cell_idx
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        image = Image.open(self.paths[i]).convert('RGB')
        pixels = self.processor(images=image, return_tensors='pt')['pixel_values']
        return pixels.squeeze(0), self.cell_idx[i]


def main():
    from transformers import CLIPProcessor
    from config import CLIP_MODEL, METADATA_PATH_OSV, IMAGE_PATH_OSV
    from energy.train import pick_device, load_grid

    argp = argparse.ArgumentParser(description='Stage C encoder finetuning.')
    argp.add_argument('--init-from', required=True,
                      help='Stage A/B checkpoint for the towers.')
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
    args = argp.parse_args()

    device = pick_device()
    latlngs_np, resolution, _ = load_grid(args.grid, want_rasters=False)

    state = torch.load(args.init_from, map_location=device)
    coarse_args = state['args']
    raster_table = None
    if coarse_args.get('rasters'):
        _, _, raster_table = load_grid(args.grid, want_rasters=True)
    model = EnergyModel(in_dim=1024, d=coarse_args['d'],
                        n_masks=coarse_args['masks'], raster_table=raster_table,
                        use_season=coarse_args.get('season', False))
    model.load_state_dict(state['model'])

    encoder = CLIPEncoder(CLIP_MODEL)
    embeddings_mod, blocks, final_norm = encoder.parts
    groups = encoder_param_groups(embeddings_mod, blocks, final_norm,
                                  args.unfreeze_blocks, args.encoder_lr,
                                  args.llrd)
    n_trainable = sum(p.numel() for g in groups for p in g['params'])
    logger.info(f'Encoder: {args.unfreeze_blocks} blocks unfrozen '
                f'({n_trainable / 1e6:.1f}M params).')

    meta = pd.read_csv(args.metadata, dtype={'id': str})
    meta = meta[meta['selection'] == 'train']
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

    def save(epoch, mean_loss):
        logger.info(f'epoch {epoch}: mean nll {mean_loss:.3f}')
        torch.save({'model': model.state_dict(),
                    'encoder': encoder.state_dict(),
                    'args': {**coarse_args, **vars(args)}, 'epoch': epoch},
                   os.path.join(args.out, f'{args.run_name}.pt'))

    train_stage_c(encoder, model, loader, grid_rff, groups,
                  epochs=args.epochs, tower_lr=args.tower_lr, device=device,
                  on_epoch_end=save)
    logger.info('Stage C complete. Rebuild the embedding cache with this '
                'encoder before Stage D/E.')


if __name__ == '__main__':
    main()
