"""PIGEON-lite — the same-data, same-encoder control row.

Reimplements PIGEON's core mechanism at a fraction of the reproduction cost:
a geocell classifier over NAIVE geocells (recursive binary box splits on the
longer axis, the algorithm in dataset_creation/geocell/naive_cell.py) trained
with haversine-smoothed soft labels (the exact formula from
preprocessing/utils.py: q ∝ exp(-(d - d_min)/tau), tau = 65 for PIGEOTTO),
on the shared cached embeddings. Skips semantic geocells and OPTICS — those
belong to the full tier-0c reproduction.

Prediction is the argmax cell's centroid (mean of its training points, as in
NaiveCell.centroid). One deliberate deviation, noted for the paper: the soft
targets are normalized so the loss is a proper cross-entropy; the original
feeds unnormalized q to the loss, which scales it by sum(q) but leaves the
gradient direction unchanged.

Usage:
    python -m energy.pigeon_lite --cache data/energy/cache \
        [--max-cell 2000] [--min-cell 1000] [--tau 65] [--epochs 20]
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
import torch
from torch import nn

from energy.losses import haversine_km_torch
from energy.evaluation import distance_metrics

logger = logging.getLogger('energy.pigeon_lite')
logging.basicConfig(level=logging.INFO)


def naive_geocells(latlngs: np.ndarray, max_cell: int=2000, min_cell: int=1000):
    """Recursive binary box splitting (naive_cell.py's mechanism).

    A cell splits on its longer axis (lat vs lng range) at the midpoint while
    it holds more than max_cell points AND both halves would keep at least
    min_cell points; otherwise it stays.

    Args:
        latlngs (np.ndarray): training coordinates [N, 2] (lat, lng)
        max_cell (int, optional): split cells larger than this.
        min_cell (int, optional): never create a cell smaller than this.

    Returns:
        tuple: (assignment np.int64 [N], centroids np.ndarray [C, 2])
    """
    assignment = np.zeros(len(latlngs), dtype=np.int64)
    centroids = []
    stack = [np.arange(len(latlngs))]

    while stack:
        rows = stack.pop()
        pts = latlngs[rows]
        can_split = len(rows) > max_cell

        if can_split:
            lat_range = pts[:, 0].max() - pts[:, 0].min()
            lng_range = pts[:, 1].max() - pts[:, 1].min()
            axis = 0 if lat_range > lng_range else 1
            thresh = (pts[:, axis].max() + pts[:, axis].min()) / 2
            left = pts[:, axis] < thresh
            if min(left.sum(), (~left).sum()) >= min_cell:
                stack.append(rows[left])
                stack.append(rows[~left])
                continue

        cell_id = len(centroids)
        assignment[rows] = cell_id
        centroids.append(pts.mean(axis=0))

    return assignment, np.array(centroids, dtype=np.float64)


class GeocellClassifier(nn.Module):
    """Their prediction-head shape: MLP over the frozen embedding -> cells."""

    def __init__(self, in_dim: int, n_cells: int, hidden: int=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, n_cells))

    def forward(self, f):
        return self.net(f)


def smoothed_targets(target_latlng: torch.Tensor, centroids: torch.Tensor,
                     tau: float) -> torch.Tensor:
    """q ∝ exp(-(d - d_min)/tau), normalized. [B, C]."""
    dist = haversine_km_torch(target_latlng, centroids)
    return torch.softmax(-(dist - dist.min(dim=1, keepdim=True).values) / tau, dim=1)


def train_pigeon_lite(embeddings, latlngs: np.ndarray, train_rows: np.ndarray,
                      val_rows: np.ndarray, max_cell: int=2000,
                      min_cell: int=1000, tau: float=65.0, epochs: int=20,
                      batch_size: int=256, lr: float=1e-3, device: str='cpu',
                      hard_labels: bool=False, seed: int=330):
    """Trains the control classifier; returns (model, centroids, history).

    hard_labels=True disables smoothing — the within-control ablation of
    their prior (mirrors A3 from the other side).
    """
    train_ll = latlngs[train_rows]
    assignment, centroids_np = naive_geocells(train_ll, max_cell, min_cell)
    n_cells = len(centroids_np)
    logger.info(f'Naive geocells: {n_cells} cells '
                f'(sizes {np.bincount(assignment).min()}-{np.bincount(assignment).max()}).')

    centroids = torch.from_numpy(centroids_np).float().to(device)
    model = GeocellClassifier(embeddings.shape[1], n_cells).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * max(1, len(train_rows) // batch_size))

    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_rows))
        epoch_loss, n_seen = 0.0, 0
        for start in range(0, len(order), batch_size):
            sel = order[start:start + batch_size]
            f = torch.from_numpy(np.asarray(embeddings[train_rows[sel]],
                                            dtype=np.float32)).to(device)
            logits = model(f)

            if hard_labels:
                targets = torch.from_numpy(assignment[sel]).to(device)
                loss = nn.functional.cross_entropy(logits, targets)
            else:
                y = torch.from_numpy(train_ll[sel]).float().to(device)
                q = smoothed_targets(y, centroids, tau)
                loss = -(q * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            epoch_loss += loss.item() * len(sel)
            n_seen += len(sel)

        metrics = evaluate_pigeon_lite(model, embeddings, latlngs, val_rows,
                                       centroids_np, device=device)
        record = {'epoch': epoch, 'train_loss': epoch_loss / n_seen, **metrics}
        history.append(record)
        logger.info(json.dumps(record))

    return model, centroids_np, history


@torch.no_grad()
def evaluate_pigeon_lite(model, embeddings, latlngs: np.ndarray,
                         rows: np.ndarray, centroids_np: np.ndarray,
                         batch_size: int=1024, device: str='cpu') -> dict:
    model.eval()
    preds = []
    for start in range(0, len(rows), batch_size):
        f = torch.from_numpy(np.asarray(embeddings[rows[start:start + batch_size]],
                                        dtype=np.float32)).to(device)
        preds.append(model(f).argmax(dim=1).cpu().numpy())
    pred_latlng = centroids_np[np.concatenate(preds)]
    return distance_metrics(pred_latlng, latlngs[rows])


def main():
    from energy.train import pick_device, load_cache

    argp = argparse.ArgumentParser(description='PIGEON-lite control.')
    argp.add_argument('--cache', default='data/energy/cache')
    argp.add_argument('--out', default='saved_models/energy')
    argp.add_argument('--run-name', default='pigeon_lite')
    argp.add_argument('--max-cell', type=int, default=2000)
    argp.add_argument('--min-cell', type=int, default=1000)
    argp.add_argument('--tau', type=float, default=65.0)
    argp.add_argument('--hard-labels', action='store_true', default=False)
    argp.add_argument('--epochs', type=int, default=20)
    argp.add_argument('--batch-size', type=int, default=256)
    argp.add_argument('--lr', type=float, default=1e-3)
    args = argp.parse_args()

    device = pick_device()
    embeddings, index = load_cache(args.cache)
    latlngs = index[['lat', 'lng']].values.astype(np.float32)
    splits = {name: np.flatnonzero((index['selection'] == name).values)
              for name in ['train', 'val']}

    model, centroids, history = train_pigeon_lite(
        embeddings, latlngs, splits['train'], splits['val'],
        max_cell=args.max_cell, min_cell=args.min_cell, tau=args.tau,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device=device, hard_labels=args.hard_labels)

    os.makedirs(args.out, exist_ok=True)
    torch.save({'model': model.state_dict(), 'centroids': centroids,
                'args': vars(args)},
               os.path.join(args.out, f'{args.run_name}.pt'))
    with open(os.path.join(args.out, f'{args.run_name}_history.json'), 'w') as fh:
        json.dump(history, fh, indent=2)


if __name__ == '__main__':
    main()
