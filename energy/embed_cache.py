"""Builds the shared frozen-encoder embedding cache from the OSV-5M metadata
CSV — the one expensive pass that serves Stage A, ablation 0a', and (via a
later converter that adds geocell labels) the PIGEON head retrain.

Output layout under --out (default data/energy/cache):
    embeddings.f16.npy   float16 memmap [N, 1024], mean-pooled CLIP hidden state
                         (matches models/clip_embedder.py CLIPEmbedding exactly)
    index.csv            row-aligned: id, lat, lng, month, selection,
                         climate_zone, drive_side

Resumable: rows are written in order with a progress marker; rerunning
continues from the last completed batch.

Usage:
    PIGEON_CLIP_MODEL=geolocal/StreetCLIP python -m energy.embed_cache \
        [--batch-size 256] [--out data/energy/cache]
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
from config import CLIP_MODEL, CLIP_EMBED_DIM, METADATA_PATH_OSV, IMAGE_PATH_OSV

logger = logging.getLogger('energy.embed_cache')
logging.basicConfig(level=logging.INFO)


def pick_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths, processor):
        self.paths = paths
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image
        image = Image.open(self.paths[idx]).convert('RGB')
        pixels = self.processor(images=image, return_tensors='pt')['pixel_values']
        return pixels.squeeze(0)


def main():
    from transformers import CLIPProcessor, CLIPVisionModel

    argp = argparse.ArgumentParser(description='Build the embedding cache.')
    argp.add_argument('--out', default='data/energy/cache')
    argp.add_argument('--batch-size', type=int, default=256)
    argp.add_argument('--num-workers', type=int, default=8)
    argp.add_argument('--metadata', default=METADATA_PATH_OSV)
    argp.add_argument('--images', default=IMAGE_PATH_OSV)
    args = argp.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = pick_device()
    logger.info(f'Encoder: {CLIP_MODEL} on {device}.')

    meta = pd.read_csv(args.metadata, dtype={'id': str})
    keep = ['id', 'lat', 'lng', 'month', 'selection']
    keep += [c for c in ['climate_zone', 'drive_side'] if c in meta.columns]
    meta[keep].to_csv(os.path.join(args.out, 'index.csv'), index=False)

    n = len(meta)
    emb_path = os.path.join(args.out, 'embeddings.f16.npy')
    marker_path = os.path.join(args.out, 'progress.json')

    embeddings = np.lib.format.open_memmap(
        emb_path, mode='r+' if os.path.exists(emb_path) else 'w+',
        dtype=np.float16, shape=(n, CLIP_EMBED_DIM))

    start_row = 0
    if os.path.exists(marker_path):
        with open(marker_path) as fh:
            start_row = json.load(fh)['rows_done']
        logger.info(f'Resuming from row {start_row}/{n}.')

    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model = CLIPVisionModel.from_pretrained(CLIP_MODEL).to(device).eval()

    paths = [os.path.join(args.images, p) for p in meta['image'].values[start_row:]]
    dataset = ImageDataset(paths, processor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         num_workers=args.num_workers)

    row = start_row
    with torch.no_grad():
        for pixels in loader:
            pixels = pixels.to(device)
            # Mean-pooled last hidden state — matches CLIPEmbedding._get_embedding
            out = model.base_model(pixel_values=pixels).last_hidden_state.mean(dim=1)
            batch = out.shape[0]
            embeddings[row:row + batch] = out.cpu().numpy().astype(np.float16)
            row += batch

            if (row - start_row) % (args.batch_size * 50) == 0:
                embeddings.flush()
                with open(marker_path, 'w') as fh:
                    json.dump({'rows_done': row}, fh)
                logger.info(f'{row}/{n} embedded.')

    embeddings.flush()
    with open(marker_path, 'w') as fh:
        json.dump({'rows_done': row}, fh)
    logger.info(f'Done: {row}/{n} embeddings at {emb_path}.')


if __name__ == '__main__':
    main()
