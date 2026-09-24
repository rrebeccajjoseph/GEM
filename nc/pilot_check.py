"""Go / no-go for the full --continuous matrix, from the ~1M-row trial runs.

The cell-free stack has only passed toy tests (nc/test_continuous_unit.py,
nc/test_infer_unit.py). Its two known failure modes both look fine at toy
scale and could come back at full scale:
  - spikes: the model hides probability mass between quadrature nodes and
    the importance weights collapse onto a few of them (ESS -> 1);
  - rough landscape: gradient refinement stalls in Fourier-feature ripples
    and moves predictions away from the best node instead of toward a mode.

Checks, each against the matched cell trial (same rows, seed, epochs):
  1. every loss finite, both runs completed
  2. ESS p10 stays above --min-ess in every epoch (no spike collapse)
  3. cont val median_km <= --max-ratio x the cell trial's
  4. on the OSV-5M test benchmark: refined median_km <= coarse (best node)
     median_km, i.e. refinement does not hurt

Writes saved_models/energy/pilot_check.json; exit 0 = go, 1 = no-go, so
the queue skips every full --continuous run on a no-go.
"""
import json, math, sys, argparse

argp = argparse.ArgumentParser()
argp.add_argument('--cont', default='pilot_cont_b_maps_gate')
argp.add_argument('--cell', default='pilot_cell_b_maps')
argp.add_argument('--bench', default='saved_models/energy/benchmark_osv5m_test_pilot_cont_b_maps_gate.json')
argp.add_argument('--min-ess', type=float, default=5.0)
argp.add_argument('--max-ratio', type=float, default=1.5)
args = argp.parse_args()

def history(run):
    return json.load(open(f'saved_models/energy/{run}_history.json'))

checks, info = {}, {}
cont, cell = history(args.cont), history(args.cell)
finite = lambda h: all(math.isfinite(r['train_loss']) and math.isfinite(r.get('median_km', math.nan)) for r in h)
checks['finite'] = bool(cont) and bool(cell) and finite(cont) and finite(cell)

ess = [r['ess_p10_min'] for r in cont if 'ess_p10_min' in r]
info['ess_p10_min_per_epoch'] = ess
checks['no_spike_collapse'] = bool(ess) and min(ess) >= args.min_ess

best_cont = min(r['median_km'] for r in cont)
best_cell = min(r['median_km'] for r in cell)
info.update(cont_val_median_km=best_cont, cell_val_median_km=best_cell,
            ratio=best_cont / best_cell)
checks['competitive_with_cells'] = best_cont <= args.max_ratio * best_cell

b = json.load(open(args.bench))
info.update({k: b[k] for k in b if k.endswith(('median_km', 'acc_1km', 'acc_25km'))})
info['refined_mean_energy_drop'] = b.get('refined_mean_energy_drop')
checks['refinement_does_not_hurt'] = (math.isfinite(b['refined_median_km'])
                                      and b['refined_median_km'] <= b['coarse_median_km'])

go = all(checks.values())
out = {'go': go, 'checks': checks, **info}
json.dump(out, open('saved_models/energy/pilot_check.json', 'w'), indent=2)
print(json.dumps(out, indent=2))
sys.exit(0 if go else 1)
