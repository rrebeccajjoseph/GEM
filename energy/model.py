"""Towers and the energy function.

E(x, y, m, s) = E_vis(x, y, m) + E_season(x, s, y) + sum_k lambda_k d_k(a_k(x), A_k(y))

The visual term is bilinear so the full grid energy map is one matmul and
Z is exact. Nothing couples m and s, so the (m, s) marginalization factorizes:

F(x, y) = -logsumexp_j(-E_vis) - logsumexp_t(-E_season) + raster terms

and no [B, M*T, |G|] tensor ever exists — only [B, M, |G|] and [B, T, |G|].

Sign convention throughout: the code works in LOGITS (= negative energies),
so F is computed as -logsumexp(vis_logits over m) - logsumexp(season_logits
over t) - raster_logits, and p(y|x) = softmax_g(-F).
"""

import math
import torch
import numpy as np
from torch import nn, Tensor

EARTH_RADIUS_KM = 6371.0


def latlng_to_unit_sphere(latlng: Tensor) -> Tensor:
    """Converts degrees [N, 2] (lat, lng) to unit 3-vectors [N, 3]."""
    lat = torch.deg2rad(latlng[:, 0])
    lng = torch.deg2rad(latlng[:, 1])
    return torch.stack([torch.cos(lat) * torch.cos(lng),
                        torch.cos(lat) * torch.sin(lng),
                        torch.sin(lat)], dim=-1)


class FourierFeatures(nn.Module):
    """Random Fourier features over the unit sphere.

    Directions are random unit 3-vectors scaled by log-spaced frequencies
    sigma in [sigma_min, sigma_max]; features are [sin, cos] pairs. Fixed
    (buffers, not parameters) so the encoding is deterministic given the seed.

    Args:
        n_freqs (int): number of frequencies (output dim = 2 * n_freqs)
        sigma_min (float): lowest frequency scale
        sigma_max (float): highest frequency scale
        seed (int): RNG seed for the directions
    """

    def __init__(self, n_freqs: int=256, sigma_min: float=1.0,
                 sigma_max: float=512.0, seed: int=330):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        directions = torch.randn(n_freqs, 3, generator=gen)
        directions = directions / directions.norm(dim=-1, keepdim=True)
        sigmas = torch.logspace(math.log10(sigma_min), math.log10(sigma_max), n_freqs)
        self.register_buffer('freqs', directions * sigmas.unsqueeze(-1))  # [F, 3]
        self.out_dim = 2 * n_freqs

    def forward(self, latlng: Tensor) -> Tensor:
        x = latlng_to_unit_sphere(latlng)          # [N, 3]
        proj = 2 * math.pi * (x @ self.freqs.T)    # [N, F]
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


def mlp(dims, final_activation: bool=False) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or final_activation:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class LocationTower(nn.Module):
    """phi(y): RFF -> MLP -> R^d. The RFF of a fixed grid can be precomputed
    once (encode_features) since grid centroids never move; only the MLP
    reruns per step — and it must rerun IN-GRAPH every step (see plan:
    detaching the grid zeroes the negative-phase gradient for this tower).
    """

    def __init__(self, d: int=512, hidden: int=512, n_freqs: int=256):
        super().__init__()
        self.rff = FourierFeatures(n_freqs=n_freqs)
        self.net = mlp([self.rff.out_dim, hidden, hidden, d])

    def encode_features(self, latlng: Tensor) -> Tensor:
        """The fixed RFF encoding (precompute for the grid, no grad needed)."""
        with torch.no_grad():
            return self.rff(latlng)

    def forward_features(self, feats: Tensor) -> Tensor:
        """MLP over precomputed RFF features — in-graph."""
        return self.net(feats)

    def forward(self, latlng: Tensor) -> Tensor:
        return self.net(self.rff(latlng))


class ImageTower(nn.Module):
    """g(f(x) ⊙ σ(mask_j)) for each of M masks -> [B, M, d].

    Gates initialize near 1.0 with small noise so masks start near-identical
    and differentiate under load-balancing pressure (Stage D).
    """

    def __init__(self, in_dim: int=1024, d: int=512, hidden: int=512,
                 n_masks: int=1, gate_init_logit: float=4.0, seed: int=330):
        super().__init__()
        self.n_masks = n_masks
        gen = torch.Generator().manual_seed(seed)
        noise = 0.01 * torch.randn(n_masks, in_dim, generator=gen)
        self.mask_logits = nn.Parameter(gate_init_logit + noise)
        self.net = mlp([in_dim, hidden, d])

    def gates(self) -> Tensor:
        return torch.sigmoid(self.mask_logits)  # [M, in_dim]

    def forward(self, f: Tensor) -> Tensor:
        gated = f.unsqueeze(1) * self.gates().unsqueeze(0)  # [B, M, in_dim]
        return self.net(gated)                              # [B, M, d]


class GaussianHead(nn.Module):
    """a_k(x) = (mu, logvar); logit = -NLL against the raster value A_k(y).

    Student-t (df degrees of freedom) when heavy_tailed, for population
    density. lambda_k is fixed at 1 wherever variance is learned (the plan's
    identifiability fix), so no lambda parameter here.
    """

    def __init__(self, in_dim: int, heavy_tailed: bool=False, df: float=4.0):
        super().__init__()
        self.head = nn.Linear(in_dim, 2)
        self.heavy_tailed = heavy_tailed
        self.df = df

    def forward(self, f: Tensor, values: Tensor, valid: Tensor) -> Tensor:
        """
        Args:
            f: image embeddings [B, in_dim]
            values: raster values at grid cells [G] (normalized, NaN-free)
            valid: bool [G], False where the raster had no data (ocean)

        Returns:
            logits [B, G]: -NLL where valid, 0 where invalid (term drops out)
        """
        out = self.head(f)                       # [B, 2]
        mu, logvar = out[:, :1], out[:, 1:]      # [B, 1]
        z2 = (values.unsqueeze(0) - mu) ** 2 / logvar.exp()  # [B, G]

        if self.heavy_tailed:
            nll = 0.5 * logvar + 0.5 * (self.df + 1) * torch.log1p(z2 / self.df)
        else:
            nll = 0.5 * (logvar + z2)

        return torch.where(valid.unsqueeze(0), -nll, torch.zeros_like(nll))


class ClimateHead(nn.Module):
    """Softmax over Köppen classes; logit = lambda * log a(x)[A(y)]."""

    def __init__(self, in_dim: int, n_classes: int=30):
        super().__init__()
        self.head = nn.Linear(in_dim, n_classes)
        # softplus-parameterized precision; learnable because the CE term has
        # no learned variance to trade off against (identifiable)
        self.log_lambda = nn.Parameter(torch.zeros(()))

    def forward(self, f: Tensor, class_idx: Tensor, valid: Tensor) -> Tensor:
        log_probs = torch.log_softmax(self.head(f), dim=-1)     # [B, C]
        idx = class_idx.clamp(min=0).unsqueeze(0).expand(f.shape[0], -1)  # [B, G]
        picked = torch.gather(log_probs, 1, idx)                # [B, G]
        lam = nn.functional.softplus(self.log_lambda)
        return torch.where(valid.unsqueeze(0), lam * picked, torch.zeros_like(picked))


class DriveSideHead(nn.Module):
    """Sigmoid over drive side; logit = -lambda * BCE against A(y)."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.head = nn.Linear(in_dim, 1)
        self.log_lambda = nn.Parameter(torch.zeros(()))

    def forward(self, f: Tensor, values: Tensor, valid: Tensor) -> Tensor:
        logit = self.head(f)                                    # [B, 1]
        bce = nn.functional.binary_cross_entropy_with_logits(
            logit.expand(-1, values.shape[0]),
            values.unsqueeze(0).expand(f.shape[0], -1), reduction='none')
        return torch.where(valid.unsqueeze(0), -bce, torch.zeros_like(bce))


class RasterBank(nn.Module):
    """All raster compatibility terms, evaluated against the [G, K] table.

    Continuous rasters are normalized to zero-mean unit-var over valid (land)
    cells at load time; log1p is applied to precip and popdens first.
    """

    LOG1P = {'precip', 'popdens'}
    HEAVY = {'popdens'}

    def __init__(self, table: dict, in_dim: int=1024):
        """
        Args:
            table (dict): name -> np.ndarray [G] raw raster values (NaN = no data)
            in_dim (int): image embedding dim
        """
        super().__init__()
        self.names = sorted(table.keys())
        self.heads = nn.ModuleDict()

        for name in self.names:
            raw = np.asarray(table[name], dtype=np.float64).copy()
            valid = ~np.isnan(raw)

            if name == 'climate':
                cls = np.where(valid, raw, 0).astype(np.int64)
                self.register_buffer(f'values_{name}', torch.from_numpy(cls))
                self.heads[name] = ClimateHead(in_dim, n_classes=30)
            elif name == 'drive_side':
                vals = np.where(valid, raw, 0.0).astype(np.float32)
                self.register_buffer(f'values_{name}', torch.from_numpy(vals))
                self.heads[name] = DriveSideHead(in_dim)
            else:
                if name in self.LOG1P:
                    raw[valid] = np.log1p(np.maximum(raw[valid], 0.0))
                mean, std = raw[valid].mean(), raw[valid].std() + 1e-8
                vals = np.where(valid, (raw - mean) / std, 0.0).astype(np.float32)
                self.register_buffer(f'values_{name}', torch.from_numpy(vals))
                self.heads[name] = GaussianHead(in_dim, heavy_tailed=name in self.HEAVY)

            self.register_buffer(f'valid_{name}', torch.from_numpy(valid))

    def forward(self, f: Tensor, grid_slice: slice=None) -> Tensor:
        """Summed raster logits [B, G(slice)]."""
        total = None
        for name in self.names:
            values = getattr(self, f'values_{name}')
            valid = getattr(self, f'valid_{name}')
            if grid_slice is not None:
                values, valid = values[grid_slice], valid[grid_slice]
            term = self.heads[name](f, values, valid)
            total = term if total is None else total + term
        return total


class SeasonHead(nn.Module):
    """E_season(f(x), s, A_clim(y)) — low-rank plausibility of a month at a
    climate. logits[b, t, g] = <u(f_b) * e_t, c_emb[A_clim(g)]> / sqrt(ds).
    """

    def __init__(self, in_dim: int=1024, ds: int=64, n_months: int=12,
                 n_climates: int=30):
        super().__init__()
        self.proj = nn.Linear(in_dim, ds)
        self.month_emb = nn.Embedding(n_months, ds)
        self.climate_emb = nn.Embedding(n_climates, ds)
        self.ds = ds

    def forward(self, f: Tensor, climate_idx: Tensor, valid: Tensor) -> Tensor:
        u = self.proj(f)                                        # [B, ds]
        ut = u.unsqueeze(1) * self.month_emb.weight.unsqueeze(0)  # [B, T, ds]
        c = self.climate_emb(climate_idx.clamp(min=0))          # [G, ds]
        logits = torch.einsum('btd,gd->btg', ut, c) / math.sqrt(self.ds)
        return torch.where(valid.unsqueeze(0).unsqueeze(0), logits,
                           torch.zeros_like(logits))            # [B, T, G]


class EnergyModel(nn.Module):
    """The full coarse model. Stage flags: rasters/season default off (Stage A).

    Args:
        in_dim (int): cached image embedding dim (1024 for CLIP ViT-L/14)
        d (int): shared bilinear dim
        n_masks (int): M (1 for Stages A-C)
        raster_table (dict, optional): name -> [G] values; enables raster terms
        use_season (bool, optional): enables the season latent (needs climate raster)
    """

    def __init__(self, in_dim: int=1024, d: int=512, n_masks: int=1,
                 raster_table: dict=None, use_season: bool=False):
        super().__init__()
        self.d = d
        self.image_tower = ImageTower(in_dim=in_dim, d=d, n_masks=n_masks)
        self.location_tower = LocationTower(d=d)
        self.rasters = RasterBank(raster_table, in_dim) if raster_table else None
        self.season = None
        if use_season:
            assert raster_table is not None and 'climate' in raster_table, \
                'Season head conditions on the climate raster.'
            self.season = SeasonHead(in_dim=in_dim)

    def vis_logits(self, img_emb: Tensor, loc_emb: Tensor) -> Tensor:
        """[B, M, d] x [G, d] -> [B, M, G] logits (= -E_vis)."""
        return torch.einsum('bmd,gd->bmg', img_emb, loc_emb) / math.sqrt(self.d)

    def neg_free_energy(self, f: Tensor, loc_emb: Tensor,
                        grid_slice: slice=None) -> Tensor:
        """-F(x, y) over (a slice of) the grid: [B, G(slice)].

        p(y|x) = softmax over the grid of this quantity. The uniform priors
        over m and s contribute constants absorbed by normalization.
        """
        img_emb = self.image_tower(f)                       # [B, M, d]
        logits = self.vis_logits(img_emb, loc_emb)          # [B, M, G]
        neg_f = torch.logsumexp(logits, dim=1)              # marginalize m

        if self.rasters is not None:
            neg_f = neg_f + self.rasters(f, grid_slice)

        if self.season is not None:
            climate_idx = self.rasters.values_climate
            climate_valid = self.rasters.valid_climate
            if grid_slice is not None:
                climate_idx = climate_idx[grid_slice]
                climate_valid = climate_valid[grid_slice]
            s_logits = self.season(f, climate_idx, climate_valid)  # [B, T, G]
            neg_f = neg_f + torch.logsumexp(s_logits, dim=1) - math.log(s_logits.shape[1])

        return neg_f

    def routing_posterior(self, f: Tensor, loc_emb_at_y: Tensor) -> Tensor:
        """r(m | x, y*) at the target locations: [B, M]. loc_emb_at_y is [B, d]."""
        img_emb = self.image_tower(f)                                   # [B, M, d]
        logits = torch.einsum('bmd,bd->bm', img_emb, loc_emb_at_y) / math.sqrt(self.d)
        return torch.softmax(logits, dim=-1)
