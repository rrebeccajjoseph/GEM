"""Randomized importance-sampled quadrature on the sphere — the cell-free
replacement for summing over H3 centroids.

Z(x) = ∫_S² exp(-F(x, y)) dy is estimated from two kinds of nodes each step:

  - N Fibonacci-lattice nodes under a fresh, uniformly random rotation
    (for any fixed node y_n, R y_n is uniform on the sphere), and
  - local nodes drawn from von Mises-Fisher kernels around every target in
    the batch, at a few scales (default 2 / 20 / 200 km),

combined with the balance-heuristic estimator over the whole mixture q:

    Ẑ = (1 / N_total) Σ_i exp(-F(x, y_i)) / q(y_i),
    q(y) = (N / N_total) / 4π + Σ_{b,s} (k / N_total) vMF(y; y*_b, κ_s)

which is unbiased for Z. Why the local part is not optional: with uniform
nodes alone the estimate is unbiased but its variance explodes for a field
sharper than the node spacing, and log Ẑ is biased low (Jensen) — so the
model learns spikes between nodes at the targets and the loss collapses
while the true Z runs away (nc/test_continuous_unit.py reproduces this:
280 nats of hidden mass). With kernels centred on the targets, the one
place a spike pays off is also the place that is always sampled. Spikes
below the smallest kernel scale stay exploitable in principle; the RFF
location encoding cannot resolve much below ~12 km anyway, and the train
loop logs the effective sample size of the weights as the tripwire.
"""

import math
import torch
import numpy as np
from torch import Tensor

from energy.model import latlng_to_unit_sphere, EARTH_RADIUS_KM


def fibonacci_sphere(n: int) -> Tensor:
    """n near-uniform unit 3-vectors (golden-angle spiral) [n, 3], float64."""
    i = torch.arange(n, dtype=torch.float64) + 0.5
    z = 1.0 - 2.0 * i / n
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    theta = math.pi * (3.0 - math.sqrt(5.0)) * i
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=-1)


def random_rotation(rng: np.random.Generator) -> Tensor:
    """A Haar-uniform rotation matrix [3, 3], float64, from a numpy RNG (so
    it is covered by the trainer's checkpointed RNG state on --resume)."""
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.sign(np.diag(r))       # makes the QR factor Haar-distributed
    if np.linalg.det(q) < 0:          # O(3) -> SO(3)
        q[:, 0] = -q[:, 0]
    return torch.from_numpy(q)


def unit_to_latlng(v: Tensor) -> Tensor:
    """Unit 3-vectors [N, 3] -> degrees [N, 2] (lat, lng)."""
    lat = torch.rad2deg(torch.asin(torch.clamp(v[:, 2], -1.0, 1.0)))
    lng = torch.rad2deg(torch.atan2(v[:, 1], v[:, 0]))
    return torch.stack([lat, lng], dim=-1)


def sample_vmf(mu: np.ndarray, kappa: float, rng: np.random.Generator) -> np.ndarray:
    """One von Mises-Fisher sample on S² per mean direction.

    Uses the closed-form inverse CDF of w = mu·y that exists on S²:
    w = 1 + log(u + (1 - u) e^{-2κ}) / κ.

    Args:
        mu (np.ndarray): unit mean directions [M, 3] float64
        kappa (float): concentration (≈ 1 / σ² for angular std σ radians)

    Returns:
        np.ndarray: unit vectors [M, 3] float64
    """
    M = mu.shape[0]
    u = rng.random(M)
    w = 1.0 + np.log(u + (1.0 - u) * np.exp(-2.0 * kappa)) / kappa
    # a uniformly random tangent direction at each mu
    t = rng.standard_normal((M, 3))
    t -= (t * mu).sum(1, keepdims=True) * mu
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return w[:, None] * mu + np.sqrt(np.clip(1.0 - w * w, 0.0, None))[:, None] * t


def log_vmf(y: Tensor, mu: Tensor, kappa: float) -> Tensor:
    """log vMF density per steradian of every y [N, 3] under every mu [M, 3]
    -> [N, M]. Uses the chord distance (1 - mu·y = |y - mu|² / 2) so float32
    stays accurate at kilometre scales, where 1 - mu·y itself would round
    away."""
    chord2 = ((y.unsqueeze(1) - mu.unsqueeze(0)) ** 2).sum(-1)
    log_norm = math.log(kappa) - math.log(2 * math.pi) - math.log1p(-math.exp(-2 * kappa))
    return log_norm - 0.5 * kappa * chord2


class SphereQuadrature:
    """The node set: fixed (evaluation) or a fresh importance-sampled draw
    (training).

    Args:
        n (int): uniform nodes (288,122 matches the res-4 grid's compute)
        device (str): where the nodes live
        local_scales_km (tuple): vMF kernel scales around each target
        local_per_target (int): local samples per target per scale
    """

    def __init__(self, n: int, device: str='cpu',
                 local_scales_km: tuple=(2.0, 20.0, 200.0), local_per_target: int=4):
        self.n = n
        self.device = device
        self.kappas = [(EARTH_RADIUS_KM / s) ** 2 for s in local_scales_km]
        self.local_per_target = local_per_target
        # float64 on the CPU (MPS has no float64); rotating 288k 3-vectors
        # there is ~ms per step, and only the float32 lat/lngs move over.
        self.base = fibonacci_sphere(n)                        # [n, 3] float64

    def fixed(self) -> Tensor:
        """The unrotated nodes as lat/lng [n, 2] float32 — deterministic, for
        evaluation and inference."""
        return unit_to_latlng(self.base).float().to(self.device)

    def rotated(self, rng: np.random.Generator) -> Tensor:
        """A fresh random rotation of the uniform nodes, lat/lng [n, 2]."""
        R = random_rotation(rng)
        return unit_to_latlng(self.base @ R.T).float().to(self.device)

    def sample(self, rng: np.random.Generator, targets: Tensor, chunk: int=8192) -> tuple:
        """One training draw: rotated uniform nodes plus local vMF nodes
        around every target, with the mixture log-density of each.

        Args:
            rng: numpy RNG (the trainer's, so --resume replays it)
            targets (Tensor): batch targets lat/lng [B, 2]

        Returns:
            tuple: (latlng [N_total, 2] float32, log_q [N_total] float32 per
                steradian), both on self.device
        """
        R = random_rotation(rng)
        uniform = self.base @ R.T                                          # [n, 3]
        mu = latlng_to_unit_sphere(targets.detach().cpu().double()).numpy()  # [B, 3]
        mu_rep = np.repeat(mu, self.local_per_target, axis=0)
        local = [sample_vmf(mu_rep, k, rng) for k in self.kappas]
        nodes = torch.cat([uniform] + [torch.from_numpy(l) for l in local])  # [N_total, 3]
        n_total = nodes.shape[0]

        # balance-heuristic mixture density, evaluated on device in float32
        nodes_d = nodes.float().to(self.device)
        mu_d = torch.from_numpy(mu).float().to(self.device)
        log_w_uniform = math.log(self.n / n_total) - math.log(4 * math.pi)
        log_w_local = math.log(self.local_per_target / n_total)
        log_q = []
        for start in range(0, n_total, chunk):
            y = nodes_d[start:start + chunk]
            terms = [torch.full((y.shape[0], 1), log_w_uniform, device=self.device)]
            for k in self.kappas:
                terms.append(log_w_local + log_vmf(y, mu_d, k))           # [Nc, B]
            log_q.append(torch.logsumexp(torch.cat(terms, dim=1), dim=1))
        latlng = unit_to_latlng(nodes).float().to(self.device)
        return latlng, torch.cat(log_q)
