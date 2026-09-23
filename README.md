# spherical-pigeon

*Consider a spherical pigeon in a vacuum.*

Energy-based geolocalization: a single exactly-normalized energy field over an
H3 grid on the sphere (288,122 res-4 cells), replacing the geocell classifier +
haversine label smoothing + auxiliary heads + OPTICS retrieval pipeline of
PIGEON (Haas et al., CVPR 2024) with:

- a bilinear visual energy (so the partition function is exact — one matmul
  over the grid, no contrastive divergence, no MCMC),
- raster compatibility terms (Köppen climate, elevation, temperature,
  precipitation, population density, drive side) that shape **inference**, not
  just representations,
- latent explanation masks and a season latent, marginalized in closed form
  (the two logsumexps factorize),
- a top-K refinement stage (an expressive scorer over H3 res-8 candidates)
  replacing retrieval.

**Cell-free mode (`--continuous`).** The energy is a continuous function of
lat/lng, and the H3 grid was only ever the normalization support. With
`--continuous`, training drops it entirely. Targets are exact coordinates,
not snapped centroids, and raster terms are sampled from lat/lng fields
(`energy.fields`). Z becomes an unbiased importance-sampled integral: each
step uses a randomly rotated Fibonacci lattice plus von Mises-Fisher nodes
around the batch's targets (`energy.quadrature`). Z is then estimated, not
exact. The local nodes are required, not a tuning option: with uniform
nodes alone the model hides unbounded mass in spikes between them
(`nc/test_continuous_unit.py` reproduces both cases). The train loss is the
log-density relative to uniform, so it goes negative. `--gate` adds a
closed-form visibility latent per raster term, so a clue the image doesn't
show stops voting.

Training data: [OSV-5M](https://huggingface.co/datasets/osv5m/osv5m)
(Astruc et al., CVPR 2024). Encoder:
[geolocal/StreetCLIP](https://huggingface.co/geolocal/StreetCLIP) by default.

**This repository contains no PIGEON code.** Baseline comparisons against
PIGEON (the tier-0c retrain and the `--osv` integration) run inside a separate
clone of the [official PIGEON release](https://github.com/LukasHaas/PIGEON),
which is licensed CC BY-NC 4.0 for academic validation. `energy/pigeon_lite.py`
here is an independent reimplementation of the *mechanisms* (naive geocells,
haversine-smoothed soft labels) used as a same-data control.

## Pipeline

```
sh get_rasters.sh                      # Köppen + GHSL rasters
python osv5m.py all                    # download OSV-5M, build metadata CSV
python -m energy.grid --out data/energy/grid.npz \
    --worldclim-tavg ... --worldclim-prec ... --elevation ... \
    --fields-out data/energy/fields.npz   # lat/lng fields for off-grid lookups
PIGEON_CLIP_MODEL=geolocal/StreetCLIP python -m energy.embed_cache
python -m energy.train --run-name stage_a                 # Stage A
python -m energy.train --run-name 0a_prime --contrastive  # ablation 0a'
python -m energy.train --run-name a3 --smooth-tau 65      # ablation A3
python -m energy.train --run-name stage_b --rasters --init-from saved_models/energy/stage_a.pt
python -m energy.train --run-name stage_b_cont --rasters --gate --continuous \
    --fields data/energy/fields.npz --init-from saved_models/energy/stage_a.pt  # cell-free B
python -m energy.finetune_encoder --init-from saved_models/energy/stage_b.pt  # Stage C
python -m energy.train --run-name stage_d --rasters --masks 16 --season \
    --checkpoint-chunks --init-from saved_models/energy/stage_c.pt
python -m energy.refine --coarse saved_models/energy/stage_d.pt   # Stage E
python -m energy.pigeon_lite                              # control row
python -m energy.benchmark --coarse ... --refiner ... --benchmark im2gps3k
```

`energy.train` writes two checkpoints per run: `{run_name}.pt` is the best
epoch by validation median_km (this is what `--init-from` and eval should
use), and `{run_name}_last.pt` is the final epoch, kept for inspecting
divergence.

`{run_name}_last.pt` also carries the full training state (optimizer, LR
scheduler, epoch, step, RNG, history). Pass `--resume
saved_models/energy/{run_name}_last.pt` to continue a run where it left
off, skipping completed epochs — this is how a job preempted on a
`*-preempt` partition picks back up after Slurm requeues it. A missing
`--resume` path just starts a fresh run, so a requeued job can always pass
the flag unconditionally.

`energy.refine` (Stage E) follows the same convention: `{run_name}.pt` is the
best epoch by val median_km on a 2k-row val subset (what `energy.benchmark
--refiner` should load), `{run_name}_last.pt` carries the full state and is
also rewritten every `--ckpt-every` steps (default 5000), so `--resume`
continues mid-epoch. Its mined top-K (`topk{K}_{hash}.npz`) is keyed by the
coarse checkpoint and shared across runs, and the dense H3 child table
(`data/energy/children_r4_to_r8_*.npy`, ~11 GB, built once with
multiprocessing) is shared by every Stage E run and by the benchmark path.
W&B gets a step-level curve (`train/loss_step`, `train/acc_sampled`) plus
per-epoch val metrics under `epoch/`, with the frozen coarse baseline on the
same rows under `coarse/` in the run summary.
