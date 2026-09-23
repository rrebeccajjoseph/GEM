"""Cell-free inference: gradient descent on the energy with respect to the
coordinate itself.

The coarse posterior over a fixed node set (evaluation.predict) is only a
starting point. Every candidate then moves continuously on the sphere to a
local minimum of F(x, y), and the lowest-energy end point is the
prediction. This replaces Stage E's H3 res-8 candidate cells: no cell
structure enters at any point.

Mechanics:
  - Starts: the top nodes of the coarse posterior, thinned by greedy
    non-maximum suppression so the K starts cover K different modes rather
    than K neighbours of one.
  - Local search: each start jumps to the best of S von Mises-Fisher
    samples at the node-spacing scale around it. Descent alone is not
    enough: a learned field is not guaranteed smooth between nodes, and a
    start a node spacing from its mode often sits in a ripple of the
    Fourier features and never gets out (nc/test_infer_unit.py).
  - Parameterization: unit 3-vectors, renormalized after each step — no pole
    singularity, no longitude wrap to special-case.
  - Gradients reach y through the location tower's Fourier features and
    through the bilinear continuous rasters (RasterFields). Categorical
    rasters sample nearest, so they are piecewise constant in y and only
    decide which basin wins, not where inside it the point settles.
  - Adam with a cosine-decayed step in radians: the first steps move
    ~node-spacing distances, the last ones sub-kilometre.
"""

import math
import numpy as np
import torch
from torch import Tensor

from energy.model import latlng_to_unit_sphere
from energy.quadrature import unit_to_latlng, sample_vmf
from energy.evaluation import predict
from energy.losses import haversine_km_torch


def nms_starts(log_p: Tensor, node_latlng: Tensor, k: int, min_sep_km: float=150.0,
               pool: int=None) -> Tensor:
    """Greedy NMS over the coarse posterior: K well-separated start points.

    Args:
        log_p (Tensor): posterior over nodes [B, N]
        node_latlng (Tensor): nodes [N, 2]
        k (int): starts per image
        min_sep_km (float): minimum distance between two starts
        pool (int): candidates considered per image (default 8k)

    Returns:
        Tensor: start coordinates [B, k, 2] (a row repeats its best start
            if fewer than k candidates survive suppression)
    """
    pool = min(pool or 8 * k, log_p.shape[1])
    top = torch.topk(log_p, pool, dim=1).indices                       # [B, P]
    starts = []
    for b in range(log_p.shape[0]):
        cand = node_latlng[top[b]]                                     # [P, 2]
        dist = haversine_km_torch(cand, cand)                          # [P, P]
        alive = torch.ones(pool, dtype=torch.bool, device=cand.device)
        chosen = []
        for i in range(pool):
            if not alive[i]:
                continue
            chosen.append(i)
            if len(chosen) == k:
                break
            alive &= dist[i] >= min_sep_km
        chosen += [chosen[0]] * (k - len(chosen))
        starts.append(cand[chosen])
    return torch.stack(starts)


def paired_neg_f(model, f: Tensor, latlng: Tensor, fields=None, group: int=16) -> Tensor:
    """-F(x_b, y_bp) for each image with its own P points: [B, P].

    neg_free_energy scores every image against every point, so images go
    in groups of `group`: a [g, g*P] map whose diagonal blocks are kept,
    bounding the wasted work at g x instead of B x.
    """
    B, P, _ = latlng.shape
    out = []
    for s in range(0, B, group):
        fg, lg = f[s:s + group], latlng[s:s + group]
        g = fg.shape[0]
        flat = lg.reshape(g * P, 2)
        raster_values = None
        if model.rasters is not None:
            raster_values = model.rasters.at_points(fields, flat)
        neg_f = model.neg_free_energy(fg, model.location_tower(flat),
                                      raster_values=raster_values)
        idx = torch.arange(g, device=f.device)
        out.append(neg_f.reshape(g, g, P)[idx, idx])                      # [g, P]
    return torch.cat(out)


@torch.no_grad()
def local_search(model, f: Tensor, starts: Tensor, fields=None, scale_km: float=50.0,
                 n: int=256, seed: int=330) -> Tensor:
    """Moves each start to the best of n vMF samples (angular std scale_km)
    around it, or keeps it if none is better: [B, K, 2]."""
    from energy.model import EARTH_RADIUS_KM
    B, K, _ = starts.shape
    mu = latlng_to_unit_sphere(starts.reshape(B * K, 2).cpu().double()).numpy()
    rng = np.random.default_rng(seed)
    pts = sample_vmf(np.repeat(mu, n, axis=0), (EARTH_RADIUS_KM / scale_km) ** 2, rng)
    cand = unit_to_latlng(torch.from_numpy(pts)).float().to(starts.device)
    cand = torch.cat([starts.reshape(B, K, 1, 2), cand.reshape(B, K, n, 2)], dim=2)
    neg_f = paired_neg_f(model, f, cand.reshape(B, K * (n + 1), 2), fields)
    best = neg_f.reshape(B, K, n + 1).argmax(dim=2)                         # [B, K]
    return torch.gather(cand, 2, best[..., None, None].expand(B, K, 1, 2)).squeeze(2)


def refine_coordinates(model, f: Tensor, starts: Tensor, fields=None, steps: int=100,
                       lr: float=2e-3, lr_min: float=2e-5) -> tuple:
    """Gradient descent on F(x, y) over y, from K starts per image.

    Args:
        model (EnergyModel): the (frozen) model
        f (Tensor): image embeddings [B, in_dim]
        starts (Tensor): start coordinates [B, K, 2]
        fields (RasterFields, optional): required iff the model has rasters
        steps (int): optimizer steps
        lr (float): initial step in radians (2e-3 ≈ 13 km)
        lr_min (float): final step in radians (2e-5 ≈ 130 m)

    Returns:
        tuple: (latlng [B, K, 2], neg_f [B, K]) at the end points
    """
    B, K, _ = starts.shape
    v = latlng_to_unit_sphere(starts.reshape(B * K, 2)).reshape(B, K, 3)
    v = v.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([v], lr=lr)
    was_training = model.training
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        for step in range(steps):
            for g in opt.param_groups:
                g['lr'] = lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * step / steps))
            u = v / v.norm(dim=-1, keepdim=True)
            latlng = unit_to_latlng(u.reshape(B * K, 3)).reshape(B, K, 2)
            energy = -paired_neg_f(model, f, latlng, fields)
            opt.zero_grad()
            energy.sum().backward()
            opt.step()
            with torch.no_grad():
                v /= v.norm(dim=-1, keepdim=True)
        with torch.no_grad():
            latlng = unit_to_latlng(v.reshape(B * K, 3)).reshape(B, K, 2)
            neg_f = paired_neg_f(model, f, latlng, fields)
    finally:
        for p in model.parameters():
            p.requires_grad_(True)
        model.train(was_training)
    return latlng.detach(), neg_f


def predict_continuous(model, f: Tensor, node_rff: Tensor, node_latlng: Tensor,
                       fields=None, k: int=8, steps: int=100, min_sep_km: float=None,
                       search_samples: int=256, **refine_kwargs) -> dict:
    """Coarse posterior over the nodes, then K NMS-separated starts, a local
    search at the node-spacing scale around each, and gradient descent; the
    lowest-energy end point is the prediction. min_sep_km defaults to three
    node spacings.

    Returns:
        dict: node_pred [B, 2] (best node, before refinement), pred [B, 2]
            (after), log_p [B, N] coarse posterior (torch), start/end neg_f
    """
    from energy.model import EARTH_RADIUS_KM
    spacing_km = EARTH_RADIUS_KM * math.sqrt(4 * math.pi / node_latlng.shape[0])
    log_p, pred_idx = predict(model, f, node_rff, grid_latlng=node_latlng, fields=fields)
    starts = nms_starts(log_p, node_latlng, k, min_sep_km=min_sep_km or 3 * spacing_km)
    if search_samples:
        starts = local_search(model, f, starts, fields, scale_km=spacing_km / 2,
                              n=search_samples)
    with torch.no_grad():
        start_neg_f = paired_neg_f(model, f, starts, fields)
    with torch.enable_grad():
        end_latlng, end_neg_f = refine_coordinates(model, f, starts, fields,
                                                   steps=steps, **refine_kwargs)
    # Adam can overshoot on a rough landscape; never trade a start for a
    # worse end point, so refinement cannot lose to the node argmax.
    worse = end_neg_f < start_neg_f
    end_latlng = torch.where(worse.unsqueeze(-1), starts, end_latlng)
    end_neg_f = torch.where(worse, start_neg_f, end_neg_f)
    best = end_neg_f.argmax(dim=1)
    rows = torch.arange(f.shape[0], device=f.device)
    return {'node_pred': node_latlng[torch.as_tensor(pred_idx, device=f.device)],
            'pred': end_latlng[rows, best], 'log_p': log_p,
            'start_neg_f': start_neg_f, 'end_neg_f': end_neg_f}


def evaluate_continuous(model, embeddings: Tensor, latlngs: np.ndarray, node_rff: Tensor,
                        node_latlng: Tensor, fields=None, k: int=8, steps: int=100,
                        batch_size: int=256, device: str='cpu', **refine_kwargs) -> dict:
    """The metric suite for a cell-free model: coarse_* (best node — the
    analogue of the cell argmax, with entropy calibration) and refined_*
    (after gradient descent), plus how much the descent lowered the energy.
    """
    from energy.evaluation import distance_metrics, calibration_metrics

    model.eval()
    node_pred, pred, entropy, gain = [], [], [], []
    for start in range(0, len(embeddings), batch_size):
        f = embeddings[start:start + batch_size].to(device)
        out = predict_continuous(model, f, node_rff, node_latlng, fields=fields, k=k,
                                 steps=steps, **refine_kwargs)
        log_p = out['log_p']
        entropy.append((-(log_p.exp() * log_p).sum(dim=1)).cpu().numpy())
        node_pred.append(out['node_pred'].cpu().numpy())
        pred.append(out['pred'].cpu().numpy())
        gain.append((out['end_neg_f'].max(1).values - out['start_neg_f'].max(1).values)
                    .cpu().numpy())

    node_pred, pred = np.concatenate(node_pred), np.concatenate(pred)
    metrics = {f'coarse_{k_}': v for k_, v in distance_metrics(node_pred, latlngs).items()}
    metrics.update({f'coarse_{k_}': v for k_, v in
                    calibration_metrics(np.concatenate(entropy), node_pred, latlngs).items()})
    metrics.update({f'refined_{k_}': v for k_, v in distance_metrics(pred, latlngs).items()})
    metrics['refined_mean_energy_drop'] = float(np.concatenate(gain).mean())
    return metrics
