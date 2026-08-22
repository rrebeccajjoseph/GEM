"""Metrics for the energy model: the standard distance thresholds PIGEON
reports (1 / 25 / 200 / 750 / 2500 km + median geodesic error) plus the
calibration diagnostics the energy formulation is meant to buy.
"""

import numpy as np
import torch
from torch import Tensor

THRESHOLDS_KM = [1, 25, 200, 750, 2500]
EARTH_RADIUS_KM = 6371.0


def haversine_km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic distance between [N, 2] arrays of (lat, lng) degrees."""
    lat1, lng1 = np.radians(a[:, 0]), np.radians(a[:, 1])
    lat2, lng2 = np.radians(b[:, 0]), np.radians(b[:, 1])
    dlat, dlng = lat2 - lat1, lng2 - lng1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


@torch.no_grad()
def predict(model, f: Tensor, grid_rff: Tensor, chunk_size: int=65536):
    """Full posterior over the grid.

    Returns:
        tuple: (log_p [B, G] torch, pred_idx [B] np.ndarray)
    """
    neg_f = []
    for start in range(0, grid_rff.shape[0], chunk_size):
        loc_emb = model.location_tower.forward_features(grid_rff[start:start + chunk_size])
        neg_f.append(model.neg_free_energy(
            f, loc_emb, grid_slice=slice(start, start + loc_emb.shape[0])))
    neg_f = torch.cat(neg_f, dim=1)                          # [B, G]
    log_p = neg_f - torch.logsumexp(neg_f, dim=1, keepdim=True)
    return log_p, log_p.argmax(dim=1).cpu().numpy()


def distance_metrics(pred_latlng: np.ndarray, true_latlng: np.ndarray) -> dict:
    """Standard PIGEON-style report."""
    dists = haversine_km(pred_latlng, true_latlng)
    out = {f'acc_{t}km': float((dists <= t).mean()) for t in THRESHOLDS_KM}
    out['median_km'] = float(np.median(dists))
    out['mean_km'] = float(np.mean(dists))
    return out


def calibration_metrics(log_p: Tensor, pred_latlng: np.ndarray,
                        true_latlng: np.ndarray, n_bins: int=10) -> dict:
    """Entropy-binned calibration: does predictive entropy rank error?

    Reports mean geodesic error per entropy bin and the Spearman rank
    correlation between per-sample entropy and error.
    """
    from scipy.stats import spearmanr

    entropy = (-(log_p.exp() * log_p).sum(dim=1)).cpu().numpy()
    dists = haversine_km(pred_latlng, true_latlng)

    order = np.argsort(entropy)
    bins = np.array_split(order, n_bins)
    bin_err = [float(dists[b].mean()) for b in bins if len(b) > 0]
    bin_ent = [float(entropy[b].mean()) for b in bins if len(b) > 0]

    rho, _ = spearmanr(entropy, dists)
    return {'spearman_entropy_error': float(rho),
            'bin_entropy': bin_ent, 'bin_error_km': bin_err}


def evaluate_on_cache(model, embeddings: Tensor, latlngs: np.ndarray,
                      grid_rff: Tensor, grid_latlngs: np.ndarray,
                      batch_size: int=256, device: str='cpu') -> dict:
    """Runs the full metric suite over a cached-embedding split."""
    model.eval()
    all_logp, all_pred = [], []
    for start in range(0, len(embeddings), batch_size):
        f = embeddings[start:start + batch_size].to(device)
        log_p, pred_idx = predict(model, f, grid_rff)
        all_logp.append(log_p.cpu())
        all_pred.append(pred_idx)

    log_p = torch.cat(all_logp, dim=0)
    pred_idx = np.concatenate(all_pred)
    pred_latlng = grid_latlngs[pred_idx]

    metrics = distance_metrics(pred_latlng, latlngs)
    metrics.update(calibration_metrics(log_p, pred_latlng, latlngs))
    return metrics
