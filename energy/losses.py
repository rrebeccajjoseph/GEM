"""Exact NLL over the grid with a streaming logsumexp, routing regularizers,
and the InfoNCE baseline objective (ablation row 0a': same towers, GeoCLIP-
style contrastive loss, same data).

The grid axis is processed in chunks. IMPORTANT: chunking alone does not
reduce autograd peak memory (every chunk's activations stay in the graph);
pass checkpoint_chunks=True to recompute per-chunk activations in the
backward pass — needed once M > 1 at |G| = 288k.
"""

import math
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint


def exact_nll(model, f: Tensor, target_idx: Tensor, grid_rff: Tensor,
              chunk_size: int=32768, checkpoint_chunks: bool=False):
    """Exact negative log-likelihood: F(x, y*) + log Z(x).

    The location tower runs over the FULL grid in-graph every call — the
    negative-phase gradient for the location tower flows through log Z, and
    detaching or caching the grid embeddings silently removes it.

    y* is ALWAYS the snapped cell centroid. Scoring the positive phase at the
    exact coordinate while Z sums over centroids makes the objective
    unbounded below (the target point is outside the normalization support,
    so the model can spike -F there indefinitely) — training diverges within
    a few hundred steps. Sub-cell placement belongs to Stage E, not here.

    Args:
        model (EnergyModel): the model
        f (Tensor): cached image embeddings [B, in_dim]
        target_idx (Tensor): grid cell index of y* [B]
        grid_rff (Tensor): precomputed RFF features of the grid [G, F]
            (fixed encoding — legitimate to precompute; the MLP on top is not)
        chunk_size (int, optional): grid cells per chunk.
        checkpoint_chunks (bool, optional): gradient-checkpoint each chunk.

    Returns:
        tuple: (nll [B], neg_f_target [B], log_z [B])
    """
    G = grid_rff.shape[0]

    def chunk_lse(rff_chunk: Tensor, start: int):
        loc_emb = model.location_tower.forward_features(rff_chunk)      # [Gc, d]
        neg_f = model.neg_free_energy(f, loc_emb,
                                      grid_slice=slice(start, start + rff_chunk.shape[0]))
        return torch.logsumexp(neg_f, dim=1)                            # [B]

    chunk_lses = []
    for start in range(0, G, chunk_size):
        rff_chunk = grid_rff[start:start + chunk_size]
        if checkpoint_chunks and torch.is_grad_enabled():
            lse = checkpoint(chunk_lse, rff_chunk, start, use_reentrant=False)
        else:
            lse = chunk_lse(rff_chunk, start)
        chunk_lses.append(lse)

    log_z = torch.logsumexp(torch.stack(chunk_lses, dim=0), dim=0)      # [B]

    # -F at the target cell centroid (see docstring: never the exact point)
    loc_emb_t = model.location_tower.forward_features(grid_rff[target_idx])  # [B, d]

    img_emb = model.image_tower(f)                                      # [B, M, d]
    vis_t = torch.einsum('bmd,bd->bm', img_emb, loc_emb_t) / (model.d ** 0.5)
    neg_f_t = torch.logsumexp(vis_t, dim=1)                             # [B]

    if model.rasters is not None:
        raster_all = model.rasters(f)                                   # [B, G]
        neg_f_t = neg_f_t + torch.gather(raster_all, 1, target_idx.unsqueeze(1)).squeeze(1)

    if model.season is not None:
        s_logits = model.season(f, model.rasters.values_climate,
                                model.rasters.valid_climate)            # [B, T, G]
        s_t = torch.gather(torch.logsumexp(s_logits, dim=1), 1,
                           target_idx.unsqueeze(1)).squeeze(1)
        neg_f_t = neg_f_t + s_t - math.log(s_logits.shape[1])

    nll = log_z - neg_f_t
    return nll, neg_f_t, log_z


def routing_regularizers(r: Tensor, gates: Tensor):
    """Mask routing regularizers (Stage D).

    Args:
        r (Tensor): routing posterior r(m | x, y*) [B, M]
        gates (Tensor): sigmoid mask gates [M, in_dim]

    Returns:
        dict: confidence (per-sample entropy, to MINIMIZE), balance
            (KL(batch mean ‖ uniform)), l1 (gate sparsity)
    """
    eps = 1e-9
    per_sample_entropy = -(r * (r + eps).log()).sum(dim=-1).mean()

    mean_r = r.mean(dim=0)                                   # [M]
    uniform = torch.full_like(mean_r, 1.0 / r.shape[1])
    balance = (mean_r * ((mean_r + eps).log() - uniform.log())).sum()

    l1 = gates.abs().mean()

    return {'confidence': per_sample_entropy, 'balance': balance, 'l1': l1}


def info_nce(model, f: Tensor, target_latlng: Tensor, temperature: float=0.07):
    """GeoCLIP-style contrastive objective for ablation 0a'.

    In-batch negatives: image b against the batch's locations. Same towers,
    same data as Stage A; only the normalization differs (batch vs exact grid).

    Args:
        model (EnergyModel): must have n_masks == 1
        f (Tensor): image embeddings [B, in_dim]
        target_latlng (Tensor): true coordinates [B, 2]
        temperature (float, optional): InfoNCE temperature.

    Returns:
        Tensor: scalar loss
    """
    img_emb = model.image_tower(f).squeeze(1)                # [B, d]
    loc_emb = model.location_tower(target_latlng)            # [B, d]

    img_emb = torch.nn.functional.normalize(img_emb, dim=-1)
    loc_emb = torch.nn.functional.normalize(loc_emb, dim=-1)

    logits = img_emb @ loc_emb.T / temperature               # [B, B]
    labels = torch.arange(f.shape[0], device=f.device)
    return 0.5 * (torch.nn.functional.cross_entropy(logits, labels) +
                  torch.nn.functional.cross_entropy(logits.T, labels))
