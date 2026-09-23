"""Stage E — refinement over the top-K coarse cells (replaces OPTICS).

The bilinear coarse energy is deliberately weak so Z stays exact; this stage
recovers expressivity where exactness is no longer needed: on a truncated
candidate set.

Procedure (coarse model FROZEN throughout — training a fine head against a
moving proposal is the main way this stage fails):

  1. Mine top-K res-4 cells per image by -F(x, y), offline (cached on disk,
     keyed by the coarse checkpoint, so every Stage E run on the same coarse
     model shares it).
  2. Expand each mined cell to its H3 res-`fine_res` children
     (default res 8, ~0.86 km scale; K=20 -> 48,020 candidates. K=64 would
     give 154k — 7^4 = 2,401 children per cell, not the ~50k a naive
     estimate suggests). The children of every grid cell are tabulated once
     into a dense [G, 2401] table on disk (`data/energy/children_*.npy`) and
     served from the GPU, so candidate assembly and negative sampling are
     pure tensor ops — the per-sample Python loop of the first version was
     CPU-bound and its per-cell cache would have needed ~50 GB of RAM.
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

Checkpoints follow `energy.train`: `{run_name}.pt` is the best epoch by val
median_km (what `energy.benchmark --refiner` should load), `{run_name}_last.pt`
carries the full training state and is also written every `--ckpt-every`
steps, so a preempted/requeued job passes `--resume` and continues
mid-epoch. W&B gets a step-level curve every `--wandb-log-every` steps plus
per-epoch val metrics next to the frozen coarse baseline on the same rows.

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
        [--topk 20] [--fine-res 8] [--cand-samples 4096] \
        [--resume saved_models/energy/stage_e_last.pt] [--wandb]
"""

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)

import json
import math
import time
import hashlib
import logging
import argparse
import numpy as np
import torch
import h3
from torch import nn, Tensor

from energy.model import EnergyModel, FourierFeatures, mlp
from energy.grid import build_grid, cell_index_map, _children, _to_latlng, _to_cell

logger = logging.getLogger('energy.refine')
logging.basicConfig(level=logging.INFO)

_to_parent = h3.cell_to_parent if hasattr(h3, 'cell_to_parent') else h3.h3_to_parent


def _h3_int(cell) -> int:
    return h3.str_to_int(cell) if isinstance(cell, str) else int(cell)


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
    up to keep Z exact, affordable here because the candidate set is small.

    Optional city term: a candidate's *nearest-observed-point* city (from
    CandidateBank's gazetteer lookup, resolved at Stage E's ~0.86km
    candidate scale — a coarse 60km RasterBank grid cell is usually bigger
    than a city, so a per-cell vote there would mostly be aliasing; a
    per-candidate nearest-point city label doesn't have that problem).
    n_cities=0 (default) disables it — every embedding call with city
    codes present still costs a lookup, so existing runs that don't pass
    --city-gazetteer stay exactly as before, zero overhead."""

    def __init__(self, in_dim: int=1024, hidden: int=256, n_cities: int=0):
        super().__init__()
        self.img_proj = mlp([in_dim, hidden, hidden])
        self.loc_enc = FineLocationEncoder(hidden=hidden)
        self.n_cities = n_cities
        joint_dim = 3 * hidden
        if n_cities > 0:
            # +1: index n_cities itself is the "no nearby labeled point"
            # sentinel (see CandidateBank._build_city — every real code is
            # < n_cities, so this slot never collides with a real city).
            self.city_emb = nn.Embedding(n_cities + 1, hidden)
            joint_dim += hidden
        self.joint = mlp([joint_dim, hidden, 1])

    def forward(self, f: Tensor, cand_latlng: Tensor, cand_city: Tensor=None) -> Tensor:
        """
        Args:
            f: image embeddings [B, in_dim]
            cand_latlng: candidate coordinates [B, C, 2]
            cand_city: candidate nearest-city codes [B, C] int64, required
                iff n_cities > 0.

        Returns:
            scores [B, C]
        """
        B, C, _ = cand_latlng.shape
        u = self.img_proj(f)                                     # [B, H]
        v = self.loc_enc(cand_latlng.reshape(B * C, 2)).reshape(B, C, -1)
        u_exp = u.unsqueeze(1).expand(-1, C, -1)
        parts = [u_exp, v, u_exp * v]
        if self.n_cities > 0:
            assert cand_city is not None, 'n_cities > 0 but no cand_city given'
            parts.append(self.city_emb(cand_city))
        joint = torch.cat(parts, dim=-1)
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
              batch_size: int=1024, chunk_size: int=65536,
              device: str='cpu') -> np.ndarray:
    """Top-K coarse cells per image by -F(x, y), plus their log-posterior.

    Returns:
        tuple: (topk_idx np.int64 [N, k], topk_logp np.float32 [N, k])
    """
    coarse.eval()
    n = embeddings.shape[0]
    G = grid_rff.shape[0]
    topk_idx = np.zeros((n, k), dtype=np.int64)
    topk_logp = np.zeros((n, k), dtype=np.float32)

    # The location tower is frozen: run it over the grid once, not per batch.
    loc_all = torch.cat([coarse.location_tower.forward_features(grid_rff[gs:gs + chunk_size])
                         for gs in range(0, G, chunk_size)])

    t0 = time.time()
    for bi, start in enumerate(range(0, n, batch_size)):
        f = torch.from_numpy(np.asarray(embeddings[start:start + batch_size],
                                        dtype=np.float32)).to(device)
        neg_f = []
        for gs in range(0, G, chunk_size):
            loc = loc_all[gs:gs + chunk_size]
            neg_f.append(coarse.neg_free_energy(
                f, loc, grid_slice=slice(gs, gs + loc.shape[0])))
        neg_f = torch.cat(neg_f, dim=1)
        log_p = neg_f - torch.logsumexp(neg_f, dim=1, keepdim=True)
        vals, idx = torch.topk(log_p, k, dim=1)
        topk_idx[start:start + f.shape[0]] = idx.cpu().numpy()
        topk_logp[start:start + f.shape[0]] = vals.cpu().numpy()
        if bi % 500 == 0:
            done = start + f.shape[0]
            logger.info(f'mined top-{k} for {done}/{n} rows '
                        f'({(time.time() - t0) / 60:.1f} min).')

    return topk_idx, topk_logp


def _children_chunk(task):
    """Worker: children ids (as H3 ints) and centroids for a chunk of cells."""
    cells, fine_res, width = task
    ids = np.zeros((len(cells), width), dtype=np.uint64)
    latlng = np.full((len(cells), width, 2), np.nan, dtype=np.float32)
    for i, c in enumerate(cells):
        kids = list(_children(c, fine_res))
        ids[i, :len(kids)] = [_h3_int(kid) for kid in kids]
        latlng[i, :len(kids)] = [_to_latlng(kid) for kid in kids]
    return ids, latlng


_city_tree, _city_codes = None, None  # per-worker-process globals


def _init_city_worker(lat: np.ndarray, lng: np.ndarray, city_code: np.ndarray):
    """mp.Pool initializer: builds the gazetteer cKDTree once per worker
    process rather than once per chunk (it's the expensive part)."""
    global _city_tree, _city_codes
    from scipy.spatial import cKDTree
    _city_tree = cKDTree(np.stack([lat, lng], axis=1))
    _city_codes = city_code


def _city_lookup_chunk(latlng_chunk: np.ndarray) -> np.ndarray:
    """Worker: nearest gazetteer city code for each candidate centroid.
    Padded (NaN, non-existent child) slots get -1 — the caller remaps that
    to the "no nearby point" sentinel, since this function doesn't know
    n_cities."""
    n, W, _ = latlng_chunk.shape
    flat = latlng_chunk.reshape(-1, 2).astype(np.float64)
    valid = ~np.isnan(flat[:, 0])
    codes = np.full(len(flat), -1, dtype=np.int32)
    if valid.any():
        _, nearest = _city_tree.query(flat[valid])
        codes[valid] = _city_codes[nearest]
    return codes.reshape(n, W)


class CandidateBank:
    """Dense table of every coarse grid cell's res-`fine_res` children.

    Two memmapped arrays on disk, built once (multiprocessing, ~45 CPU-min
    for 288k cells at res 4 -> 8) and shared by every Stage E run:
      ids    uint64  [G, W]     H3 ints, 0-padded
      latlng float32 [G, W, 2]  centroids, NaN-padded
    W = 7^(fine_res - coarse_res); pentagon parents have fewer children, so
    the padding is masked wherever the table is used.

    Optional third table, city: for each candidate centroid, the nearest
    gazetteer point's city code (int32 [G, W], n_cities = "no labeled point
    nearby"). Built the same way but via cKDTree nearest-neighbor lookup
    against `city_gazetteer` (an .npz of lat/lng/city_code/categories, see
    nc/build_city_gazetteer.py) rather than H3 child enumeration — a city is
    usually smaller than even a fine (~0.86km) candidate cell, so this
    resolves per-candidate-point rather than voting per-cell.
    """

    def __init__(self, cells: list, coarse_res: int, fine_res: int,
                 table_dir: str='data/energy', n_workers: int=None,
                 city_gazetteer: str=None):
        self.cells = cells
        self.coarse_res = coarse_res
        self.fine_res = fine_res
        self.n_children = 7 ** (fine_res - coarse_res)
        prefix = os.path.join(table_dir, f'children_r{coarse_res}_to_r{fine_res}')
        self.ids_path = prefix + '_ids.npy'
        self.latlng_path = prefix + '_latlng.npy'
        if not (os.path.exists(self.ids_path) and os.path.exists(self.latlng_path)):
            self._build(n_workers)
        self.ids = np.load(self.ids_path, mmap_mode='r')
        self.latlng = np.load(self.latlng_path, mmap_mode='r')
        assert self.ids.shape == (len(cells), self.n_children), \
            f'{self.ids_path} does not match the grid ({self.ids.shape} vs {len(cells)} cells)'
        self._device_cache = {}

        self.city = None
        self.city_categories = None
        self.n_cities = 0
        if city_gazetteer:
            gz = np.load(city_gazetteer, allow_pickle=True)
            self.city_categories = list(gz['categories'])
            self.n_cities = len(self.city_categories)
            gz_key = hashlib.md5(
                f'{os.path.abspath(city_gazetteer)}:{os.path.getmtime(city_gazetteer)}'
                .encode()).hexdigest()[:10]
            city_path = f'{prefix}_city_{gz_key}.npy'
            if not os.path.exists(city_path):
                self._build_city(city_path, gz['lat'], gz['lng'], gz['city_code'], n_workers)
            self.city = np.load(city_path, mmap_mode='r')
            assert self.city.shape == self.ids.shape, \
                f'{city_path} does not match the children table {self.ids.shape}'

    def _build(self, n_workers: int=None):
        import multiprocessing as mp
        if n_workers is None:
            n_workers = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') \
                else (os.cpu_count() or 1)
            n_workers = max(1, min(32, n_workers))
        G, W = len(self.cells), self.n_children
        chunk = 1024
        tasks = [(self.cells[i:i + chunk], self.fine_res, W) for i in range(0, G, chunk)]
        logger.info(f'Building the res-{self.fine_res} child table for {G} cells '
                    f'({W} children each) with {n_workers} workers -> {self.latlng_path}')
        os.makedirs(os.path.dirname(self.ids_path) or '.', exist_ok=True)
        ids_tmp, ll_tmp = self.ids_path + '.tmp.npy', self.latlng_path + '.tmp.npy'
        ids = np.lib.format.open_memmap(ids_tmp, mode='w+', dtype=np.uint64, shape=(G, W))
        ll = np.lib.format.open_memmap(ll_tmp, mode='w+', dtype=np.float32, shape=(G, W, 2))
        t0 = time.time()
        with mp.Pool(n_workers) as pool:
            for ci, (cid, cll) in enumerate(pool.imap(_children_chunk, tasks)):
                s = ci * chunk
                ids[s:s + len(cid)] = cid
                ll[s:s + len(cll)] = cll
                if ci % 32 == 0:
                    logger.info(f'  child table: {min(s + chunk, G)}/{G} cells '
                                f'({(time.time() - t0) / 60:.1f} min)')
        ids.flush(); ll.flush()
        del ids, ll
        os.replace(ids_tmp, self.ids_path)
        os.replace(ll_tmp, self.latlng_path)
        logger.info(f'Child table built in {(time.time() - t0) / 60:.1f} min.')

    def _build_city(self, city_path: str, lat: np.ndarray, lng: np.ndarray,
                    city_code: np.ndarray, n_workers: int=None):
        import multiprocessing as mp
        if n_workers is None:
            n_workers = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') \
                else (os.cpu_count() or 1)
            n_workers = max(1, min(32, n_workers))
        G, W = self.latlng.shape[0], self.n_children
        chunk = 2048
        logger.info(f'Building the city-lookup table for {G} cells x {W} candidates '
                    f'against a {len(lat)}-point gazetteer with {n_workers} workers '
                    f'-> {city_path}')
        os.makedirs(os.path.dirname(city_path) or '.', exist_ok=True)
        tmp = city_path + '.tmp.npy'
        out = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.int32, shape=(G, W))
        chunks = [np.array(self.latlng[i:i + chunk]) for i in range(0, G, chunk)]
        t0 = time.time()
        with mp.Pool(n_workers, initializer=_init_city_worker,
                    initargs=(lat, lng, city_code)) as pool:
            for ci, codes in enumerate(pool.imap(_city_lookup_chunk, chunks)):
                s = ci * chunk
                codes[codes < 0] = self.n_cities  # padded (non-existent) child slots
                out[s:s + len(codes)] = codes
                if ci % 16 == 0:
                    logger.info(f'  city table: {min(s + chunk, G)}/{G} cells '
                                f'({(time.time() - t0) / 60:.1f} min)')
        out.flush()
        del out
        os.replace(tmp, city_path)
        logger.info(f'City table built in {(time.time() - t0) / 60:.1f} min.')

    def to_device(self, device: str):
        """(latlng [G, W, 2] float32 with NaN -> 0, valid [G, W] bool) on device."""
        if device not in self._device_cache:
            ll = torch.from_numpy(np.array(self.latlng))   # copy: memmap is read-only
            valid = ~torch.isnan(ll[..., 0])
            ll = torch.nan_to_num(ll, nan=0.0)
            self._device_cache[device] = (ll.to(device), valid.to(device))
        return self._device_cache[device]

    def to_device_city(self, device: str):
        """City-code table [G, W] int64 on device, or None if this bank was
        built without a city_gazetteer. Separate from to_device() rather
        than folded into its tuple so every existing call site (no city
        term) is untouched."""
        if self.city is None:
            return None
        key = f'city:{device}'
        if key not in self._device_cache:
            city = torch.from_numpy(np.array(self.city)).long()  # copy: memmap read-only
            self._device_cache[key] = city.to(device)
        return self._device_cache[key]


def compute_targets(latlngs_np: np.ndarray, topk_idx: np.ndarray, bank: CandidateBank,
                    idx_map: dict) -> np.ndarray:
    """Index of each row's true fine cell in its flat [K * W] candidate layout
    (slot * W + offset-within-parent), or -1 when the true coarse parent is
    not among the mined top-K (the row is dropped from Stage E training)."""
    n = len(latlngs_np)
    W = bank.n_children
    fine_int = np.zeros(n, dtype=np.uint64)
    parent_pos = np.zeros(n, dtype=np.int64)
    t0 = time.time()
    for r in range(n):
        c = _to_cell(float(latlngs_np[r, 0]), float(latlngs_np[r, 1]), bank.fine_res)
        fine_int[r] = _h3_int(c)
        parent_pos[r] = idx_map[_to_parent(c, bank.coarse_res)]
        if r % 1000000 == 0 and r:
            logger.info(f'  targets: {r}/{n} rows ({(time.time() - t0) / 60:.1f} min)')

    hit = topk_idx == parent_pos[:, None]                     # [n, K]
    covered = hit.any(axis=1)
    slot = hit.argmax(axis=1)

    offset = np.full(n, -1, dtype=np.int64)
    rows = np.flatnonzero(covered)
    rows = rows[np.argsort(parent_pos[rows], kind='stable')]
    uniq, starts = np.unique(parent_pos[rows], return_index=True)
    ends = np.append(starts[1:], len(rows))
    for p, s, e in zip(uniq, starts, ends):
        rr = rows[s:e]
        m = bank.ids[p][None, :] == fine_int[rr][:, None]     # [m, W]
        assert m.any(axis=1).all(), f'fine cell missing from child table row {p}'
        offset[rr] = m.argmax(axis=1)

    return np.where(covered, slot * W + offset, -1)


def _save_atomic(obj, path: str):
    torch.save(obj, path + '.tmp')
    os.replace(path + '.tmp', path)


def train_refiner(scorer, embeddings, topk_idx: np.ndarray, targets: np.ndarray,
                  bank: CandidateBank, topk_logp: np.ndarray=None, epochs: int=3, batch_size: int=64,
                  lr: float=3e-4, weight_decay: float=0.01, cand_samples: int=4096,
                  device: str='cpu', seed: int=330, wandb_run=None,
                  log_every: int=50, ckpt_every: int=5000, eval_fn=None,
                  out_dir: str=None, run_name: str='stage_e', resume: str=None,
                  stop_after_steps: int=None, eval_every: int=0,
                  prior_in_loss: bool=True, save_args: dict=None):
    """Trains the fine scorer with sampled cross-entropy over candidate sets.

    Per sample: the true fine cell plus (cand_samples - 1) candidates drawn
    uniformly without replacement from the mined set (sampled softmax over
    a uniform subsample — unbiased ranking target for the truncated
    normalization used at inference). With `prior_in_loss` (default) the
    logits are the same product of experts used at inference,
    log p_coarse(parent) - log n_children + score, so the scorer learns a
    residual on top of the frozen coarse posterior instead of re-learning
    coarse discrimination from scratch — training the score alone and
    multiplying the prior in afterwards let the scorer's coarse-level noise
    degrade the 25/200 km accuracy below the coarse model. Rows whose true fine cell is not
    covered by the mined top-K (targets == -1) are dropped; the drop rate is
    logged. Everything after the embedding gather runs on the GPU.

    `embeddings`, `topk_idx` and `targets` are row-aligned (training rows).

    Returns:
        dict: training stats (drop_rate, final_loss, best_epoch, best_median_km,
              history)
    """
    import torch.nn.functional as F

    scorer = scorer.to(device).train()
    opt = torch.optim.AdamW(scorer.parameters(), lr=lr, weight_decay=weight_decay)
    table_ll, table_valid = bank.to_device(device)
    table_city = bank.to_device_city(device)  # None if bank has no city_gazetteer
    W = bank.n_children
    K = topk_idx.shape[1]

    if prior_in_loss and topk_logp is None:
        raise ValueError('prior_in_loss needs topk_logp')
    keep = np.flatnonzero(targets >= 0)
    n_all, n_train = len(targets), len(keep)
    drop_rate = 1.0 - n_train / max(n_all, 1)
    logger.info(f'Stage E rows: {n_train}/{n_all} covered by the top-{K} set, '
                f'coverage drop rate {drop_rate:.3f}'
                + (' — ABOVE 15%: the coarse model is not ready for Stage E'
                   if drop_rate > 0.15 else ''))
    if wandb_run is not None:
        wandb_run.summary['drop_rate'] = drop_rate
        wandb_run.summary['train_rows'] = n_train

    steps_per_epoch = math.ceil(n_train / batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
    S = min(cand_samples, K * W)

    global_step, start_epoch, start_batch = 0, 0, 0
    history, best_metric, best_epoch = [], math.inf, -1
    if resume and os.path.exists(resume):
        ck = torch.load(resume, map_location=device)
        scorer.load_state_dict(ck['scorer'])
        opt.load_state_dict(ck['optimizer'])
        sched.load_state_dict(ck['scheduler'])
        global_step = ck['global_step']
        history = ck.get('history', [])
        best_metric = ck.get('best_median_km', math.inf)
        best_epoch = ck.get('best_epoch', -1)
        start_epoch, start_batch = ck['epoch'], ck['step_in_epoch']
        if start_batch >= steps_per_epoch:
            start_epoch, start_batch = start_epoch + 1, 0
        if ck.get('steps_per_epoch') not in (None, steps_per_epoch):
            logger.warning(f"--resume checkpoint had {ck['steps_per_epoch']} steps/epoch, "
                           f'this run has {steps_per_epoch}; the epoch permutation and '
                           f'LR schedule will not line up exactly.')
        logger.info(f'Resumed from {resume}: epoch {start_epoch}, batch {start_batch}/'
                    f'{steps_per_epoch}, global_step {global_step}, best epoch '
                    f'{best_epoch} (median_km {best_metric}).')
    elif resume:
        logger.info(f'--resume {resume} not found; starting a fresh run.')

    def save_last(epoch, step_in_epoch):
        if out_dir is None:
            return
        _save_atomic({'scorer': scorer.state_dict(), 'optimizer': opt.state_dict(),
                      'scheduler': sched.state_dict(), 'global_step': global_step,
                      'epoch': epoch, 'step_in_epoch': step_in_epoch,
                      'steps_per_epoch': steps_per_epoch, 'history': history,
                      'best_epoch': best_epoch, 'best_median_km': best_metric,
                      'drop_rate': drop_rate},
                     os.path.join(out_dir, f'{run_name}_last.pt'))

    last_loss = float('nan')
    stopped = False
    for epoch in range(start_epoch, epochs):
        scorer.train()
        # Epoch permutation is a pure function of (seed, epoch) so a resumed run
        # continues through the same order; negative sampling is reseeded per
        # (epoch, batch) for the same reason.
        order = np.random.default_rng(seed + epoch).permutation(keep)
        gen = torch.Generator(device=device)
        epoch_loss, epoch_n, epoch_correct = 0.0, 0, 0
        t_epoch = time.time()
        for b in range(start_batch, steps_per_epoch):
            rows = order[b * batch_size:(b + 1) * batch_size]
            B = len(rows)
            f = torch.from_numpy(np.asarray(embeddings[rows], dtype=np.float32)).to(device)
            tk = torch.from_numpy(topk_idx[rows]).to(device)             # [B, K]
            tgt = torch.from_numpy(targets[rows]).to(device)             # [B]
            cand = table_ll[tk].reshape(B, K * W, 2)
            valid_kw = table_valid[tk]                                    # [B, K, W]
            valid = valid_kw.reshape(B, K * W)
            city = table_city[tk].reshape(B, K * W) if table_city is not None else None
            if prior_in_loss:
                plp = torch.from_numpy(topk_logp[rows]).to(device)        # [B, K]
                prior = (plp - valid_kw.sum(dim=-1).float().log()).repeat_interleave(W, dim=1)

            # Uniform subsample without replacement, true cell forced to column 0
            gen.manual_seed(seed * 1000003 + epoch * 1000 + b)
            keys = torch.rand(B, K * W, device=device, generator=gen)
            keys.masked_fill_(~valid, -1.0)
            keys[torch.arange(B, device=device), tgt] = 2.0
            sel = keys.topk(S, dim=1).indices                            # [B, S]
            cand_s = torch.gather(cand, 1, sel.unsqueeze(-1).expand(-1, -1, 2))
            valid_s = torch.gather(valid, 1, sel)
            city_s = torch.gather(city, 1, sel) if city is not None else None

            scores = scorer(f, cand_s, cand_city=city_s)
            if prior_in_loss:
                scores = scores + torch.gather(prior, 1, sel)
            scores = scores.masked_fill(~valid_s, float('-inf'))
            loss = F.cross_entropy(scores, torch.zeros(B, dtype=torch.long, device=device))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
            opt.step()
            sched.step()
            global_step += 1

            last_loss = loss.item()
            correct = int((scores.argmax(dim=1) == 0).sum())
            epoch_loss += last_loss * B
            epoch_n += B
            epoch_correct += correct

            if wandb_run is not None and global_step % log_every == 0:
                import wandb
                wandb.log({'train/loss_step': last_loss,
                           'train/acc_sampled': correct / B,
                           'train/grad_norm': float(grad_norm),
                           'train/lr': sched.get_last_lr()[0],
                           'train/rows_per_s': epoch_n / max(time.time() - t_epoch, 1e-6),
                           'train/epoch_progress': (b + 1) / steps_per_epoch,
                           'epoch': epoch}, step=global_step)
            if (b + 1) % 1000 == 0:
                logger.info(f'epoch {epoch} batch {b + 1}/{steps_per_epoch}: '
                            f'loss {epoch_loss / epoch_n:.3f}, '
                            f'acc@sampled {epoch_correct / epoch_n:.3f}, '
                            f'{epoch_n / (time.time() - t_epoch):.0f} rows/s')
            if ckpt_every and global_step % ckpt_every == 0:
                save_last(epoch, b + 1)
            if eval_every and eval_fn is not None and global_step % eval_every == 0 \
                    and b + 1 < steps_per_epoch:
                mid = eval_fn(scorer)
                scorer.train()
                logger.info(f'step {global_step} val: ' + json.dumps(mid))
                if wandb_run is not None:
                    import wandb
                    wandb.log({f'val/{k}': v for k, v in mid.items()}, step=global_step)
            if stop_after_steps and global_step >= stop_after_steps:
                save_last(epoch, b + 1)
                logger.info(f'--stop-after-steps {stop_after_steps} reached; checkpoint '
                            f'written, exiting.')
                stopped = True
                break
        if stopped:
            break
        start_batch = 0

        record = {'epoch': epoch,
                  'train_loss': epoch_loss / max(epoch_n, 1),
                  'train_acc_sampled': epoch_correct / max(epoch_n, 1),
                  'drop_rate': drop_rate}
        if eval_fn is not None:
            record.update(eval_fn(scorer))
            scorer.train()
        history.append(record)
        logger.info(json.dumps(record))
        if wandb_run is not None:
            import wandb
            wandb.log({f'epoch/{k}': v for k, v in record.items() if k != 'epoch'},
                      step=global_step)

        cur = record.get('median_km')
        if cur is None or cur < best_metric:
            if cur is not None:
                best_metric = cur
            best_epoch = epoch
            if out_dir is not None:
                _save_atomic({'scorer': scorer.state_dict(), 'epoch': epoch,
                              'best_epoch': best_epoch, 'best_median_km': best_metric,
                              'args': save_args or {}},
                             os.path.join(out_dir, f'{run_name}.pt'))
                logger.info(f'New best checkpoint: epoch {epoch}, median_km {cur}.')
        save_last(epoch, steps_per_epoch)
        if out_dir is not None:
            hist_path = os.path.join(out_dir, f'{run_name}_history.json')
            with open(hist_path + '.tmp', 'w') as fh:
                json.dump(history, fh, indent=2)
            os.replace(hist_path + '.tmp', hist_path)

    return {'drop_rate': drop_rate, 'final_loss': last_loss, 'best_epoch': best_epoch,
            'best_median_km': best_metric, 'history': history, 'stopped_early': stopped}


@torch.no_grad()
def refine_predictions(scorer, embeddings, topk_idx: np.ndarray,
                       topk_logp: np.ndarray, bank: CandidateBank,
                       device: str='cpu', rows_per_batch: int=16):
    """Refined predictions via the truncated product of experts.

    Three decodes of the same refined posterior are returned:
      pred_latlng       hierarchical mode — the coarse parent with the most
                        aggregated refined mass, then its best child. The
                        flat argmax over all K*W children favours the
                        peakiest cell over the heaviest one, which is what the
                        distance-threshold metrics reward; this is the
                        prediction benchmark.py reports.
      pred_latlng_mean  posterior mean (on the sphere) within that parent —
                        collapses to the coarse centroid when the fine
                        posterior is flat, to the mode when it is sharp.
      pred_latlng_flat  the flat argmax, for comparison.

    Returns:
        dict: pred_latlng / pred_latlng_mean / pred_latlng_flat [N, 2],
              coverage [N] (coarse mass in the top-K set),
              fine_entropy [N] (entropy of the refined candidate posterior)
    """
    scorer = scorer.to(device).eval()
    table_ll, table_valid = bank.to_device(device)
    table_city = bank.to_device_city(device)
    W = bank.n_children
    n = embeddings.shape[0]
    K = topk_idx.shape[1]
    preds = np.zeros((n, 2), dtype=np.float32)
    preds_mean = np.zeros((n, 2), dtype=np.float32)
    preds_flat = np.zeros((n, 2), dtype=np.float32)
    coverage = np.zeros(n, dtype=np.float32)
    entropy = np.zeros(n, dtype=np.float32)

    for start in range(0, n, rows_per_batch):
        rows = slice(start, min(start + rows_per_batch, n))
        B = rows.stop - rows.start
        f = torch.from_numpy(np.asarray(embeddings[rows], dtype=np.float32)).to(device)
        tk = torch.from_numpy(topk_idx[rows]).to(device)                 # [B, K]
        cand = table_ll[tk].reshape(B, K * W, 2)
        valid = table_valid[tk].reshape(B, K * W)
        n_valid = table_valid[tk].sum(dim=-1).float()                    # [B, K]
        city = table_city[tk].reshape(B, K * W) if table_city is not None else None

        scores = scorer(f, cand, cand_city=city)                         # [B, K*W]
        parent_logp = torch.from_numpy(topk_logp[rows]).to(device)       # [B, K]
        prior = (parent_logp - n_valid.log()).repeat_interleave(W, dim=1)
        log_post = (prior + scores).masked_fill(~valid, float('-inf'))
        log_post = log_post - torch.logsumexp(log_post, dim=1, keepdim=True)

        ar = torch.arange(B, device=device)
        preds_flat[rows] = cand[ar, log_post.argmax(dim=1)].cpu().numpy()

        lp = log_post.view(B, K, W)
        parent_mass = torch.logsumexp(lp, dim=2)                          # [B, K]
        pbest = parent_mass.argmax(dim=1)                                 # [B]
        lp_p = lp[ar, pbest]                                              # [B, W]
        cand_p = cand.view(B, K, W, 2)[ar, pbest]                         # [B, W, 2]
        preds[rows] = cand_p[ar, lp_p.argmax(dim=1)].cpu().numpy()
        w = (lp_p - parent_mass[ar, pbest].unsqueeze(1)).exp()            # within-parent posterior
        w = w.masked_fill(~valid.view(B, K, W)[ar, pbest], 0.0).unsqueeze(-1)
        lat = torch.deg2rad(cand_p[..., 0]); lng = torch.deg2rad(cand_p[..., 1])
        xyz = torch.stack([lat.cos() * lng.cos(), lat.cos() * lng.sin(), lat.sin()], dim=-1)
        m = (w * xyz).sum(dim=1)                                          # [B, 3]
        preds_mean[rows] = torch.stack([
            torch.rad2deg(torch.atan2(m[:, 2], m[:, :2].norm(dim=1))),
            torch.rad2deg(torch.atan2(m[:, 1], m[:, 0]))], dim=1).cpu().numpy()

        coverage[rows] = parent_logp.exp().sum(dim=1).cpu().numpy()
        p = log_post.exp()
        entropy[rows] = (-(p * log_post.masked_fill(~valid, 0.0)).sum(dim=1)).cpu().numpy()

    return {'pred_latlng': preds, 'pred_latlng_mean': preds_mean,
            'pred_latlng_flat': preds_flat, 'coverage': coverage, 'fine_entropy': entropy}


def _decode_metrics(out: dict, true_latlng: np.ndarray) -> dict:
    """distance_metrics for the hierarchical decode plus the mean/flat variants
    (median_km, acc_1km, acc_25km, acc_200km) under mean_* / flat_* prefixes."""
    from energy.evaluation import distance_metrics
    m = distance_metrics(out['pred_latlng'], true_latlng)
    for tag in ('mean', 'flat'):
        alt = distance_metrics(out[f'pred_latlng_{tag}'], true_latlng)
        m.update({f'{tag}_{k}': alt[k] for k in ('median_km', 'acc_1km', 'acc_25km', 'acc_200km')})
    m['mean_topk_coverage'] = float(out['coverage'].mean())
    m['mean_fine_entropy'] = float(out['fine_entropy'].mean())
    return m


def main():
    from energy.train import pick_device, load_grid, load_cache
    from energy.evaluation import distance_metrics

    argp = argparse.ArgumentParser(description='Stage E refinement.')
    argp.add_argument('--coarse', required=True, help='Coarse model checkpoint.')
    argp.add_argument('--cache', default='data/energy/cache')
    argp.add_argument('--grid', default='data/energy/grid.npz')
    argp.add_argument('--out', default='saved_models/energy')
    argp.add_argument('--children-dir', default='data/energy',
                      help='Where the dense H3 child table lives (built once, shared).')
    argp.add_argument('--run-name', default='stage_e')
    argp.add_argument('--topk', type=int, default=20)
    argp.add_argument('--fine-res', type=int, default=8)
    argp.add_argument('--cand-samples', type=int, default=4096)
    argp.add_argument('--no-prior-in-loss', action='store_true', default=False,
                      help='Ablation: train the scorer on its own logits (the product with '
                           'the coarse prior is then only applied at inference).')
    argp.add_argument('--hidden', type=int, default=256,
                      help='Scorer width (image proj, fine location encoder, joint MLP).')
    argp.add_argument('--city-gazetteer', default=None,
                      help='Path to a city_gazetteer.npz (nc/build_city_gazetteer.py). '
                           'When given, the scorer gets each candidate\'s nearest-'
                           'observed-point city as an extra feature — a city is usually '
                           'smaller than even a fine (~0.86km) candidate cell, so this '
                           'resolves it as a per-candidate nearest-point lookup rather '
                           'than a coarse RasterBank grid vote. Off by default: no '
                           'change to any existing run that doesn\'t pass this.')
    argp.add_argument('--eval-every', type=int, default=10000,
                      help='Also refine the per-epoch val rows every N steps mid-epoch '
                           '(logged under val/, no checkpoint selection).')
    argp.add_argument('--epochs', type=int, default=3)
    argp.add_argument('--batch-size', type=int, default=64)
    argp.add_argument('--lr', type=float, default=3e-4)
    argp.add_argument('--weight-decay', type=float, default=0.01)
    argp.add_argument('--seed', type=int, default=330)
    argp.add_argument('--train-samples', type=int, default=None,
                      help='Subsample the train split to this many rows (smoke tests).')
    argp.add_argument('--epoch-eval-samples', type=int, default=2000,
                      help='Val rows refined after every epoch (checkpoint selection).')
    argp.add_argument('--eval-samples', type=int, default=20000,
                      help='Val rows refined once at the end with the best checkpoint.')
    argp.add_argument('--num-workers', type=int, default=None,
                      help='Processes for the one-off child-table build.')
    argp.add_argument('--resume', default=None,
                      help='Resume from {run_name}_last.pt (restores scorer, optimizer, '
                           'LR schedule, epoch, mid-epoch position, history). Missing '
                           'file = start fresh, so a requeued job can always pass this.')
    argp.add_argument('--ckpt-every', type=int, default=5000,
                      help='Write {run_name}_last.pt every N optimizer steps.')
    argp.add_argument('--stop-after-steps', type=int, default=None,
                      help='Debug: checkpoint and exit after N global steps.')
    argp.add_argument('--wandb', action='store_true', default=False,
                      help='Log training curves to Weights & Biases.')
    argp.add_argument('--wandb-project', default='spherical-pigeon')
    argp.add_argument('--wandb-entity', default=None)
    argp.add_argument('--wandb-mode', default='online', choices=['online', 'offline', 'disabled'])
    argp.add_argument('--wandb-log-every', type=int, default=50,
                      help='Step-level W&B point every N optimizer steps.')
    argp.add_argument('--wandb-id-salt', default='',
                      help='Appended to run_name when deriving the deterministic W&B run id '
                           '(W&B refuses to reuse a deleted id; bump this to start over).')
    args = argp.parse_args()

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
        wandb_id = hashlib.md5((args.run_name + args.wandb_id_salt).encode()).hexdigest()[:16]
        wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                               name=args.run_name, id=wandb_id, resume='allow',
                               config=vars(args))

    device = pick_device()
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    logger.info(f'Device: {device}.')
    latlngs_np, resolution, _ = load_grid(args.grid, want_rasters=False)
    embeddings, index = load_cache(args.cache)
    cells, _ = build_grid(resolution)
    idx_map = cell_index_map(cells)

    # Frozen coarse model, rebuilt from its checkpointed config
    state = torch.load(args.coarse, map_location=device)
    coarse_args = state['args']
    raster_table = None
    if coarse_args.get('rasters'):
        _, _, raster_table = load_grid(args.grid, want_rasters=True)
    coarse = EnergyModel(in_dim=embeddings.shape[1], d=coarse_args['d'],
                         n_masks=coarse_args['masks'], raster_table=raster_table,
                         use_season=coarse_args.get('season', False),
                         gated=coarse_args.get('gate', False)).to(device)
    coarse.load_state_dict(state['model'])
    for p in coarse.parameters():
        p.requires_grad_(False)
    coarse.eval()

    grid_rff = coarse.location_tower.encode_features(
        torch.from_numpy(latlngs_np).float().to(device))

    splits = {name: np.flatnonzero((index['selection'] == name).values)
              for name in ['train', 'val']}
    all_latlng = index[['lat', 'lng']].values.astype(np.float32)
    tr = splits['train']
    if args.train_samples is not None and args.train_samples < len(tr):
        tr = np.sort(np.random.default_rng(args.seed).choice(tr, args.train_samples,
                                                              replace=False))
    logger.info(f"Rows: train {len(tr)} (of {len(splits['train'])}), val {len(splits['val'])}.")
    os.makedirs(args.out, exist_ok=True)

    # Mined top-K, keyed by the coarse checkpoint + cache so every Stage E run on
    # the same coarse model shares it (mining 5.1M rows is minutes on an A100).
    key_src = f'{os.path.abspath(args.coarse)}:{os.path.getmtime(args.coarse)}:' \
              f'{os.path.abspath(args.cache)}:{len(index)}'
    key = hashlib.md5(key_src.encode()).hexdigest()[:10]
    mined_path = os.path.join(args.out, f'topk{args.topk}_{key}.npz')
    if os.path.exists(mined_path):
        mined = np.load(mined_path)
        topk_idx, topk_logp = mined['idx'], mined['logp']
        logger.info(f'Loaded mined top-K from {mined_path}.')
    else:
        logger.info(f'Mining top-{args.topk} cells for {len(index)} rows (offline).')
        topk_idx, topk_logp = mine_topk(coarse, embeddings, grid_rff,
                                        k=args.topk, device=device)
        np.savez(mined_path + '.tmp.npz', idx=topk_idx, logp=topk_logp)
        os.replace(mined_path + '.tmp.npz', mined_path)
        logger.info(f'Saved mined top-K to {mined_path}.')

    bank = CandidateBank(cells, resolution, args.fine_res, table_dir=args.children_dir,
                         n_workers=args.num_workers, city_gazetteer=args.city_gazetteer)
    if args.city_gazetteer:
        logger.info(f'City term enabled: {bank.n_cities} classes from {args.city_gazetteer}.')

    targets_path = os.path.join(args.out, f'topk{args.topk}_{key}_targets_r{args.fine_res}'
                                          f'_n{len(tr)}_s{args.seed}.npy')
    if os.path.exists(targets_path):
        targets = np.load(targets_path)
        logger.info(f'Loaded Stage E targets from {targets_path}.')
    else:
        logger.info(f'Locating the true res-{args.fine_res} cell for {len(tr)} training rows.')
        targets = compute_targets(all_latlng[tr], topk_idx[tr], bank, idx_map)
        np.save(targets_path + '.tmp.npy', targets)
        os.replace(targets_path + '.tmp.npy', targets_path)

    t0 = time.time()
    train_emb = np.asarray(embeddings[tr])                      # f16 in RAM, ~2 KB/row
    logger.info(f'Loaded {len(tr)} training embeddings into RAM '
                f'({train_emb.nbytes / 1e9:.1f} GB, {time.time() - t0:.0f} s).')

    # Frozen coarse baseline on the same val rows: the Stage E lift is the gap
    # between epoch/median_km and this.
    def coarse_metrics(rows):
        pred = latlngs_np[topk_idx[rows, 0]].astype(np.float32)
        m = distance_metrics(pred, all_latlng[rows])
        m['mean_topk_coverage'] = float(np.exp(topk_logp[rows]).sum(axis=1).mean())
        return m

    # index.csv's val rows are grouped by nearby GPS trace (e.g. one drive
    # sits in a contiguous block), so a *prefix* slice is geographically
    # skewed, not a fair subsample — checkpoint selection on it would pick
    # whichever epoch happens to fit that one region. Shuffle once (seeded)
    # before slicing for the per-epoch metric; the final report still uses
    # every val row so it is unaffected either way.
    val_order = np.random.default_rng(args.seed).permutation(splits['val'])
    va_epoch = val_order[:args.epoch_eval_samples]
    va_final = splits['val'][:args.eval_samples]
    coarse_epoch = coarse_metrics(va_epoch)
    logger.info('Coarse baseline on the per-epoch val rows: ' + json.dumps(coarse_epoch))
    if wandb_run is not None:
        for k, v in coarse_epoch.items():
            wandb_run.summary[f'coarse/{k}'] = v

    va_epoch_emb = np.asarray(embeddings[va_epoch])

    def eval_fn(scorer):
        out = refine_predictions(scorer, va_epoch_emb, topk_idx[va_epoch],
                                 topk_logp[va_epoch], bank, device=device)
        m = _decode_metrics(out, all_latlng[va_epoch])
        m['coarse_median_km'] = coarse_epoch['median_km']
        m['coarse_acc_1km'] = coarse_epoch['acc_1km']
        m['coarse_acc_25km'] = coarse_epoch['acc_25km']
        return m

    scorer = JointMLPScorer(in_dim=embeddings.shape[1], hidden=args.hidden,
                            n_cities=bank.n_cities)
    stats = train_refiner(scorer, train_emb, topk_idx[tr], targets, bank,
                          topk_logp=topk_logp[tr],
                          prior_in_loss=not args.no_prior_in_loss,
                          epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                          weight_decay=args.weight_decay, cand_samples=args.cand_samples,
                          device=device, seed=args.seed, wandb_run=wandb_run,
                          log_every=args.wandb_log_every, ckpt_every=args.ckpt_every,
                          eval_fn=eval_fn, out_dir=args.out, run_name=args.run_name,
                          resume=args.resume, stop_after_steps=args.stop_after_steps,
                          eval_every=args.eval_every,
                          # n_cities isn't a CLI flag (it's derived from
                          # whatever city_gazetteer resolved to at load
                          # time) — save it explicitly so benchmark.py can
                          # rebuild the scorer's embedding size without
                          # re-loading the gazetteer file.
                          save_args={**vars(args), 'n_cities': bank.n_cities})
    if stats.pop('stopped_early'):
        if wandb_run is not None:
            wandb.finish()
        return

    # Final report with the best-epoch checkpoint (what benchmark --refiner loads)
    best_path = os.path.join(args.out, f'{args.run_name}.pt')
    scorer.load_state_dict(torch.load(best_path, map_location=device)['scorer'])
    out = refine_predictions(scorer, np.asarray(embeddings[va_final]), topk_idx[va_final],
                             topk_logp[va_final], bank, device=device)
    metrics = _decode_metrics(out, all_latlng[va_final])
    metrics.update({f'coarse_{k}': v for k, v in coarse_metrics(va_final).items()})
    metrics.update({k: v for k, v in stats.items() if k != 'history'})
    metrics['eval_rows'] = int(len(va_final))
    logger.info(json.dumps(metrics))
    with open(os.path.join(args.out, f'{args.run_name}_metrics.json'), 'w') as fh:
        json.dump(metrics, fh, indent=2)
    if wandb_run is not None:
        wandb.log({f'final/{k}': v for k, v in metrics.items()
                   if not isinstance(v, (list, dict))})
        wandb.finish()


if __name__ == '__main__':
    main()
