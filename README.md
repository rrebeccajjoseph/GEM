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
    --worldclim-tavg ... --worldclim-prec ... --elevation ...
PIGEON_CLIP_MODEL=geolocal/StreetCLIP python -m energy.embed_cache
python -m energy.train --run-name stage_a                 # Stage A
python -m energy.train --run-name 0a_prime --contrastive  # ablation 0a'
python -m energy.train --run-name a3 --smooth-tau 65      # ablation A3
python -m energy.train --run-name stage_b --rasters --init-from saved_models/energy/stage_a.pt
python -m energy.finetune_encoder --init-from saved_models/energy/stage_b.pt  # Stage C
python -m energy.train --run-name stage_d --rasters --masks 16 --season \
    --checkpoint-chunks --init-from saved_models/energy/stage_c.pt
python -m energy.refine --coarse saved_models/energy/stage_d.pt   # Stage E
python -m energy.pigeon_lite                              # control row
python -m energy.benchmark --coarse ... --refiner ... --benchmark im2gps3k
```
