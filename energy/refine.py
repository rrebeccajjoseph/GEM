"""Stage E — refinement over the top-K coarse cells (replaces OPTICS).

The bilinear coarse energy is deliberately weak so Z stays exact; this stage
recovers expressivity where exactness is no longer needed: on a truncated
candidate set.

Procedure (coarse model FROZEN throughout — training a fine head against a
moving proposal is the main way this stage fails):

  1. Mine top-K res-4 cells per image by -F(x, y), offline.
  2. Expand each mined cell to its H3 res-`fine_res` children
     (default res 8, ~0.86 km scale; K=20 -> 48,020 candidates. K=64 would
     give 154k — 7^4 = 2,401 children per cell, not the ~50k a naive
     estimate suggests).
  3. Train an expressive scorer with cross-entropy over the candidate set;
     samples whose true fine cell is not covered are dropped (the drop rate
     is logged — above ~15% the coarse model isn't ready for Stage E).
  4. At inference the refined posterior over the candidate set is the
     truncated product of experts:

         log p(y) = log p_coarse(parent(y)) - log n_children + score_fine(y)

     normalized over the candidates. (The plan phrased this as SNIS with
     q = p_coarse; with that proposal the importance weights reduce to
     exactly this product — stated here without the sampling vocabulary.)
     The coarse mass covered by the top-K set is reported per sample: it is
     the truncation bias diagnostic.

Scorers:
  - JointMLPScorer (default): joint MLP over [image proj; phi_fine(y);
    elementwise product]. Runs on the cached pooled embeddings — strictly
    more expressive than the bilinear form, which is the point.
  - CrossAttentionScorer: location queries attending over image patch
    tokens. Patch tokens cannot be cached at 5.1M-image scale, so this
    variant requires re-encoding images at Stage E training time; it is
    provided for the (token-fed) upgrade path and tested with synthetic
    tokens.

Usage:
    python -m energy.refine --coarse saved_models/energy/stage_a.pt \
        --cache data/energy/cache --grid data/energy/grid.npz \
        [--topk 20] [--fine-res 8] [--cand-samples 4096]
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
import torch
from torch import nn, Tensor

from energy.model import EnergyModel, FourierFeatures, mlp
from energy.grid import build_grid, _children, _to_latlng, _to_cell

logger = logging.getLogger('energy.refine')
logging.basicConfig(level=logging.INFO)


class FineLocationEncoder(nn.Module):
    """phi_fine(y): RFF with higher max frequency than the coarse tower —
    res-8 cells are ~0.86 km apart and need finer spatial detail."""

    def __init__(self, hidden: int=256, n_freqs: int=256,
                 sigma_max: float=4096.0, seed: int=331):
        super().__init__()
        self.rff = FourierFeatures(n_freqs=n_freqs, sigma_max=sigma_max, seed=seed)
        self.net = mlp([self.rff.out_dim, hidden, hidden])
        self.out_dim = hidden

    def forward(self, latlng: Tensor) -> Tensor:
        return self.net(self.rff(latlng))


class JointMLPScorer(nn.Module):
    """score(f, y) via a joint MLP — the expressivity the bilinear form gave
    up to keep Z exact, affordable here because the candidate set is small."""

    def __init__(self, in_dim: int=1024, hidden: int=256):
        super().__init__()
        self.img_proj = mlp([in_dim, hidden, hidden])
        self.loc_enc = FineLocationEncoder(hidden=hidden)
        self.joint = mlp([3 * hidden, hidden, 1])

    def forward(self, f: Tensor, cand_latlng: Tensor) -> Tensor:
        """
        Args:
            f: image embeddings [B, in_dim]
            cand_latlng: candidate coordinates [B, C, 2]

        Returns:
            scores [B, C]
        """
        B, C, _ = cand_latlng.shape
        u = self.img_proj(f)                                     # [B, H]
        v = self.loc_enc(cand_latlng.reshape(B * C, 2)).reshape(B, C, -1)
        u_exp = u.unsqueeze(1).expand(-1, C, -1)
        joint = torch.cat([u_exp, v, u_exp * v], dim=-1)
        return self.joint(joint).squeeze(-1)


class CrossAttentionScorer(nn.Module):
    """Location queries attend over image patch tokens (upgrade path;
    requires a token provider at train/inference time)."""

    def __init__(self, token_dim: int=1024, hidden: int=256, n_heads: int=4):
        super().__init__()
        self.token_proj = nn.Linear(token_dim, hidden)
        self.loc_enc = FineLocationEncoder(hidden=hidden)
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.joint = mlp([3 * hidden, hidden, 1])

    def forward(self, tokens: Tensor, cand_latlng: Tensor) -> Tensor:
        """
        Args:
            tokens: patch tokens [B, L, token_dim]
            cand_latlng: candidate coordinates [B, C, 2]

        Returns:
            scores [B, C]
        """
        B, C, _ = cand_latlng.shape
        q = self.loc_enc(cand_latlng.reshape(B * C, 2)).reshape(B, C, -1)
        kv = self.token_proj(tokens)
        attended, _ = self.attn(q, kv, kv)                       # [B, C, H]
        joint = torch.cat([q, attended, q * attended], dim=-1)
        return self.joint(joint).squeeze(-1)


@torch.no_grad()
def mine_topk(coarse: EnergyModel, embeddings, grid_rff: Tensor, k: int,
              batch_size: int=256, chunk_size: int=65536,
              device: str='cpu') -> np.ndarray:
    """Top-K coarse cells per image by -F(x, y), plus their log-posterior.

    Returns:
        tuple: (topk_idx np.int64 [N, k], topk_logp np.float32 [N, k])
    """
    coarse.eval()
    n = embeddings.shape[0]
    topk_idx = np.zeros((n, k), dtype=np.int64)
    topk_logp = np.zeros((n, k), dtype=np.float32)

    for start in range(0, n, batch_size):
        f = torch.from_numpy(np.asarray(embeddings[start:start + batch_size],
                                        dtype=np.float32)).to(device)
        neg_f = []
        for gs in range(0, grid_rff.shape[0], chunk_size):
            loc = coarse.location_tower.forward_features(grid_rff[gs:gs + chunk_size])
            neg_f.append(coarse.neg_free_energy(
                f, loc, grid_slice=slice(gs, gs + loc.shape[0])))
        neg_f = torch.cat(neg_f, dim=1)
        log_p = neg_f - torch.logsumexp(neg_f, dim=1, keepdim=True)
        vals, idx = torch.topk(log_p, k, dim=1)
        topk_idx[start:start + f.shape[0]] = idx.cpu().numpy()
        topk_logp[start:start + f.shape[0]] = vals.cpu().numpy()

    return topk_idx, topk_logp


class CandidateBank:
    """Children coordinates of coarse cells at the fine resolution, cached.

    For each coarse grid position, stores (fine H3 ids, latlngs [n, 2]).
    """

    def __init__(self, cells: list, coarse_res: int, fine_res: int):
        self.cells = cells
        self.fine_res = fine_res
        self.n_children = 7 ** (fine_res - coarse_res)
        self._store = {}

    def get(self, coarse_pos: int):
        if coarse_pos not in self._store:
            kids = list(_children(self.cells[coarse_pos], self.fine_res))
            latlngs = np.array([_to_latlng(c) for c in kids], dtype=np.float32)
            self._store[coarse_pos] = (kids, latlngs)
        return self._store[coarse_pos]

    def candidates_for(self, topk_pos: np.ndarray):
        """All candidates for one sample's top-K coarse cells.

        Returns:
            tuple: (fine_ids list, latlngs np [C, 2],
                    parent_pos np [C] — index into topk_pos)
        """
        ids, lls, parents = [], [], []
        for j, pos in enumerate(topk_pos):
            kids, latlngs = self.get(int(pos))
            ids.extend(kids)
            lls.append(latlngs)
            parents.append(np.full(len(kids), j, dtype=np.int64))
        return ids, np.concatenate(lls), np.concatenate(parents)


def true_fine_positions(lats, lngs, fine_res: int):
    """H3 ids of the true fine cells."""
    return [_to_cell(la, ln, fine_res) for la, ln in zip(lats, lngs)]


def train_refiner(scorer, embeddings, latlngs_np: np.ndarray,
                  topk_idx: np.ndarray, bank: CandidateBank,
                  epochs: int=3, batch_size: int=64, lr: float=3e-4,
                  cand_samples: int=4096, device: str='cpu', seed: int=330):
    """Trains the fine scorer with sampled cross-entropy over candidate sets.

    Per sample: the true fine cell plus (cand_samples - 1) candidates drawn
    uniformly from the mined set (sampled softmax over a uniform subsample —
    unbiased ranking target for the truncated normalization used at
    inference). Samples whose true fine cell is not covered by the mined
    top-K are dropped; the drop rate is logged.

    Returns:
        dict: training stats (drop_rate, final_loss)
    """
    scorer = scorer.to(device).train()
    opt = torch.optim.AdamW(scorer.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    n = embeddings.shape[0]

    true_fine = true_fine_positions(latlngs_np[:, 0], latlngs_np[:, 1], bank.fine_res)

    dropped, kept, last_loss = 0, 0, float('nan')
    for epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            rows = order[start:start + batch_size]

            batch_f, batch_cands, batch_targets = [], [], []
            for r in rows:
                ids, lls, _ = bank.candidates_for(topk_idx[r])
                try:
                    t = ids.index(true_fine[r])
                except ValueError:
                    dropped += 1
                    continue
                kept += 1

                if len(ids) > cand_samples:
                    keep = rng.choice(len(ids), size=cand_samples, replace=False)
                    if t not in keep:
                        keep[0] = t
                    lls = lls[keep]
                    t = int(np.flatnonzero(keep == t)[0])

                batch_f.append(r)
                batch_cands.append(lls)
                batch_targets.append(t)

            if not batch_f:
                continue

            # Pad candidate sets to the batch max with -inf-masked dummies
            max_c = max(c.shape[0] for c in batch_cands)
            cand = np.zeros((len(batch_f), max_c, 2), dtype=np.float32)
            mask = np.zeros((len(batch_f), max_c), dtype=bool)
            for i, c in enumerate(batch_cands):
                cand[i, :c.shape[0]] = c
                mask[i, :c.shape[0]] = True

            f = torch.from_numpy(np.asarray(embeddings[batch_f],
                                            dtype=np.float32)).to(device)
            cand_t = torch.from_numpy(cand).to(device)
            mask_t = torch.from_numpy(mask).to(device)
            targets = torch.tensor(batch_targets, device=device)

            scores = scorer(f, cand_t)
            scores = scores.masked_fill(~mask_t, float('-inf'))
            loss = nn.functional.cross_entropy(scores, targets)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = loss.item()

        drop_rate = dropped / max(dropped + kept, 1)
        logger.info(f'epoch {epoch}: loss {last_loss:.3f}, '
                    f'coverage drop rate {drop_rate:.3f}'
                    + (' — ABOVE 15%: the coarse model is not ready for Stage E'
                       if drop_rate > 0.15 else ''))

    return {'drop_rate': dropped / max(dropped + kept, 1), 'final_loss': last_loss}


@torch.no_grad()
def refine_predictions(scorer, embeddings, topk_idx: np.ndarray,
                       topk_logp: np.ndarray, bank: CandidateBank,
                       device: str='cpu', score_chunk: int=8192):
    """Refined predictions via the truncated product of experts.

    Returns:
        dict: pred_latlng [N, 2], coverage [N] (coarse mass in the top-K set),
              fine_entropy [N] (entropy of the refined candidate posterior)
    """
    scorer = scorer.to(device).eval()
    n = embeddings.shape[0]
    preds = np.zeros((n, 2), dtype=np.float32)
    coverage = np.zeros(n, dtype=np.float32)
    entropy = np.zeros(n, dtype=np.float32)
    log_nc = math.log(bank.n_children)

    for r in range(n):
        _, lls, parents = bank.candidates_for(topk_idx[r])
        f = torch.from_numpy(np.asarray(embeddings[r:r + 1],
                                        dtype=np.float32)).to(device)

        scores = []
        for cs in range(0, len(lls), score_chunk):
            cand = torch.from_numpy(lls[cs:cs + score_chunk]).unsqueeze(0).to(device)
            scores.append(scorer(f, cand).squeeze(0))
        scores = torch.cat(scores)                               # [C]

        parent_logp = torch.from_numpy(topk_logp[r][parents]).to(device)
        log_post = parent_logp - log_nc + scores
        log_post = log_post - torch.logsumexp(log_post, dim=0)

        best = int(log_post.argmax())
        preds[r] = lls[best]
        coverage[r] = float(np.exp(topk_logp[r]).sum())
        p = log_post.exp()
        entropy[r] = float(-(p * log_post).sum())

    return {'pred_latlng': preds, 'coverage': coverage, 'fine_entropy': entropy}


def main():
    from energy.train import pick_device, load_grid, load_cache
    from energy.evaluation import distance_metrics

    argp = argparse.ArgumentParser(description='Stage E refinement.')
    argp.add_argument('--coarse', required=True, help='Coarse model checkpoint.')
    argp.add_argument('--cache', default='data/energy/cache')
    argp.add_argument('--grid', default='data/energy/grid.npz')
    argp.add_argument('--out', default='saved_models/energy')
    argp.add_argument('--run-name', default='stage_e')
    argp.add_argument('--topk', type=int, default=20)
    argp.add_argument('--fine-res', type=int, default=8)
    argp.add_argument('--cand-samples', type=int, default=4096)
    argp.add_argument('--epochs', type=int, default=3)
    argp.add_argument('--batch-size', type=int, default=64)
    argp.add_argument('--lr', type=float, default=3e-4)
    argp.add_argument('--eval-samples', type=int, default=20000)
    args = argp.parse_args()

    device = pick_device()
    latlngs_np, resolution, _ = load_grid(args.grid, want_rasters=False)
    embeddings, index = load_cache(args.cache)
    cells, _ = build_grid(resolution)

    # Frozen coarse model, rebuilt from its checkpointed config
    state = torch.load(args.coarse, map_location=device)
    coarse_args = state['args']
    raster_table = None
    if coarse_args.get('rasters'):
        _, _, raster_table = load_grid(args.grid, want_rasters=True)
    coarse = EnergyModel(in_dim=embeddings.shape[1], d=coarse_args['d'],
                         n_masks=coarse_args['masks'], raster_table=raster_table,
                         use_season=coarse_args.get('season', False)).to(device)
    coarse.load_state_dict(state['model'])
    for p in coarse.parameters():
        p.requires_grad_(False)

    grid_rff = coarse.location_tower.encode_features(
        torch.from_numpy(latlngs_np).float().to(device))

    splits = {name: np.flatnonzero((index['selection'] == name).values)
              for name in ['train', 'val']}
    all_latlng = index[['lat', 'lng']].values.astype(np.float32)

    mined_path = os.path.join(args.out, f'{args.run_name}_topk.npz')
    if os.path.exists(mined_path):
        mined = np.load(mined_path)
        topk_idx, topk_logp = mined['idx'], mined['logp']
        logger.info(f'Loaded mined top-K from {mined_path}.')
    else:
        logger.info(f'Mining top-{args.topk} cells for {len(index)} rows (offline).')
        topk_idx, topk_logp = mine_topk(coarse, embeddings, grid_rff,
                                        k=args.topk, device=device)
        os.makedirs(args.out, exist_ok=True)
        np.savez(mined_path, idx=topk_idx, logp=topk_logp)

    bank = CandidateBank(cells, resolution, args.fine_res)
    scorer = JointMLPScorer(in_dim=embeddings.shape[1])

    tr = splits['train']
    stats = train_refiner(scorer, embeddings[tr] if isinstance(embeddings, np.ndarray)
                          else np.asarray(embeddings[tr]),
                          all_latlng[tr], topk_idx[tr], bank,
                          epochs=args.epochs, batch_size=args.batch_size,
                          lr=args.lr, cand_samples=args.cand_samples, device=device)

    va = splits['val'][:args.eval_samples]
    out = refine_predictions(scorer, np.asarray(embeddings[va]), topk_idx[va],
                             topk_logp[va], bank, device=device)
    metrics = distance_metrics(out['pred_latlng'], all_latlng[va])
    metrics['mean_topk_coverage'] = float(out['coverage'].mean())
    metrics.update(stats)
    logger.info(json.dumps(metrics))

    os.makedirs(args.out, exist_ok=True)
    torch.save({'scorer': scorer.state_dict(), 'args': vars(args)},
               os.path.join(args.out, f'{args.run_name}.pt'))
    with open(os.path.join(args.out, f'{args.run_name}_metrics.json'), 'w') as fh:
        json.dump(metrics, fh, indent=2)


if __name__ == '__main__':
    main()
