#!/bin/bash
# The cell-free experiment matrix on one 8x MI355X National Compute node.
#
#   bash nc/cont_pipeline.sh launch    start the CPU phases + 8 GPU workers (tmux)
#   bash nc/cont_pipeline.sh status    one line per queued job: ok / failed / running / waiting
#   bash nc/cont_pipeline.sh <phase>   run one CPU phase in the foreground
#   bash nc/cont_pipeline.sh unclaim <job>   release a job whose worker crashed
#
# CPU phases (own tmux session each): dl_osv, dl_mp16, dl_bench, rasters, overlap.
# Everything on a GPU goes through one queue (jobs() below). 8 workers, one per
# GPU, each claim the first job whose deps are met (mkdir = atomic claim),
# run it, repeat. A job whose dep FAILED is marked skipped (rc 3) instead of
# being waited on forever. Every job has a data/logs/<job>.done marker
# (lib.sh), so re-running launch after a crash redoes only what did not finish.
#
# The questions the matrix answers (3 seeds per arm, identical recipes):
#   1. corpus:  OSV-5M vs OSV-5M+MP-16 vs MP-16-only, Stage A and B (maps+gate)
#   2. cells:   cell_* vs cont_* on the same data and seeds
#   3. clues:   rasters off / on / on+gate; label rasters vs real maps
#   4. fine:    Stage E refiner vs gradient descent, acc_1km / acc_25km
source /mnt/nvme/sp/spherical-pigeon/nc/lib.sh
cd $REPO
export PIGEON_CLIP_MODEL=geolocal/StreetCLIP
N=8
EPOCHS=${EPOCHS:-8}
SEEDS=(330 331 332)
PILOT_ROWS=${PILOT_ROWS:-1000000}
PILOT_EPOCHS=${PILOT_EPOCHS:-3}
OVERLAP_OUT=data/benchmarks/overlap
MAPS=data/energy/grid_maps.npz
CLAIMS=data/logs/claims
BENCHES=(osv5m_test im2gps3k yfcc4k)
mkdir -p data/logs $CLAIMS

# --- arms: name -> training args (seed and cache added per job) -----------
# Suffix = corpus: none = OSV-5M, _comb = OSV-5M + MP-16, _mp16 = MP-16 only.
ARMS=(cell_a cont_a cell_b_maps cont_b_maps cont_b_maps_gate cell_b_labels
      cont_a_comb cont_b_maps_gate_comb cont_a_mp16 cont_b_maps_gate_mp16
      cell_b_labels_mp16 cell_b_labels_comb)
arm_args() {
  case $1 in
    cell_a)            echo "--grid data/energy/grid.npz" ;;
    cont_a|cont_a_*)   echo "--continuous --grid data/energy/grid.npz" ;;
    cell_b_maps)       echo "--rasters --grid $MAPS" ;;
    cont_b_maps)       echo "--continuous --rasters --grid $MAPS" ;;
    cont_b_maps_gate*) echo "--continuous --rasters --gate --grid $MAPS" ;;
    cell_b_labels*)    echo "--rasters --grid data/energy/grid_geo.npz" ;;
  esac
}
arm_grid() { arm_args $1 | grep -oE -- '--grid [^ ]+' | cut -d' ' -f2; }
arm_cache() {
  case $1 in
    *_comb) echo data/energy/cache_combined ;;
    *_mp16) echo data/energy/cache_mp16 ;;
    *)      echo data/energy/cache ;;
  esac
}
arm_exclude() {  # the leakage list in THAT cache's id space (OSV-5M needs none)
  case $1 in
    *_comb) echo "--exclude-ids $OVERLAP_OUT/exclude_ids_combined.txt" ;;
    *_mp16) echo "--exclude-ids $OVERLAP_OUT/exclude_ids.txt" ;;
  esac
}
shards() { seq -f "cache_$1_shard%g" 0 $((N - 1)) | tr '\n' ' '; }
arm_deps() {
  local d
  case $1 in
    *_comb) d="combine overlap" ;;
    *_mp16) d="$(shards mp16)overlap" ;;
    *)      d="$(shards osv)" ;;
  esac
  case $1 in
    cell_b_labels*) d="$d geo_labels" ;;
    *_maps*)       d="$d maps" ;;
  esac
  # go / no-go: no full --continuous run starts until the ~1M-row trial passes
  [[ $1 == cont_* ]] && d="$d pilot_check"
  echo $d
}

# --- the queue, in priority order: "job|deps" -----------------------------
jobs() {
  local i a s
  for i in $(seq 0 $((N - 1))); do echo "cache_osv_shard$i|dl_osv"; done
  # the trial: cell vs --continuous on the same ~1M rows, seed, epochs and clue
  # terms (both ungated, so --continuous is the only difference; see pilot_check.py),
  # then a benchmark of the continuous one, then the go / no-go check
  echo "pilot_cell_b_maps|$(shards osv) maps"
  echo "pilot_cont_b_maps|$(shards osv) maps"
  echo "bench_encode_osv5m_test|bench_osv_csv"
  echo "pilot_bench|bench_encode_osv5m_test pilot_cont_b_maps"
  echo "pilot_check|pilot_cell_b_maps pilot_cont_b_maps pilot_bench"
  for a in "${ARMS[@]:0:6}"; do echo "${a}_s${SEEDS[0]}|$(arm_deps $a)"; done
  for i in $(seq 0 $((N - 1))); do echo "cache_mp16_shard$i|dl_mp16"; done
  echo "combine|$(shards osv)$(shards mp16)"
  for a in "${ARMS[@]:6}"; do echo "${a}_s${SEEDS[0]}|$(arm_deps $a)"; done
  for s in "${SEEDS[@]:1}"; do
    for a in "${ARMS[@]}"; do echo "${a}_s$s|$(arm_deps $a)"; done
  done
  # seeds after the first also wait on it: the first run builds the shared
  # 11 GB res-4 -> res-8 child table that the others then reuse
  echo "stage_e_s${SEEDS[0]}|cell_b_maps_s${SEEDS[0]}"
  for s in "${SEEDS[@]:1}"; do echo "stage_e_s$s|cell_b_maps_s$s stage_e_s${SEEDS[0]}"; done
  echo "bench_encode_im2gps3k|dl_bench"
  echo "bench_encode_yfcc4k|dl_bench"
  local b
  for b in "${BENCHES[@]}"; do
    for s in "${SEEDS[@]}"; do
      for a in "${ARMS[@]}"; do
        # MP-16 arms are scored on the Flickr benchmarks only after the leakage check
        local extra=""; [[ $a == *_comb || $a == *_mp16 ]] && [ $b != osv5m_test ] && extra=" overlap"
        echo "bench_${b}__${a}_s$s|bench_encode_$b ${a}_s$s$extra"
      done
      echo "bench_${b}__stage_e_s$s|bench_encode_$b stage_e_s$s"
    done
  done
}

# --- running one job ------------------------------------------------------
run_job() {  # <gpu> <job>
  local gpu=$1 job=$2 arm seed
  case $job in
    cache_*_shard*)
      local tag=${job#cache_}; tag=${tag%_shard*}; local i=${job##*_shard}
      local meta=data/osv5m/metadata_osv5m.csv images=data/osv5m/images out=data/energy/cache
      [ $tag = mp16 ] && { meta=data/mp16/metadata_mp16.csv; images=data/mp16/images; out=data/energy/cache_mp16; }
      mkdir -p $out
      # create the shared memmap under a lock: 8 concurrent open_memmap(w+) race
      ( flock 9; $PY -c "import numpy as np, pandas as pd, os; p='$out/embeddings.f16.npy'; os.path.exists(p) or np.lib.format.open_memmap(p, mode='w+', dtype=np.float16, shape=(len(pd.read_csv('$meta', usecols=['id'])), 1024))" ) 9>$out/.create.lock
      run_on_gpu $gpu $job energy.embed_cache --metadata $meta --images $images \
          --out $out --shard $i/$N --batch-size 256 --num-workers 24 ;;
    combine) run_cpu combine $PY nc/combine_caches.py ;;
    pilot_cell_b_maps|pilot_cont_b_maps)
      arm=${job#pilot_}
      run_on_gpu $gpu $job energy.train --run-name $job --seed ${SEEDS[0]} \
          --resume saved_models/energy/${job}_last.pt --cache data/energy/cache \
          $(arm_args $arm) --max-rows $PILOT_ROWS --epochs $PILOT_EPOCHS \
          --fields data/energy/fields.npz WBFLAGS ;;
    pilot_bench)
      run_on_gpu $gpu $job energy.benchmark --coarse saved_models/energy/pilot_cont_b_maps.pt \
          --benchmark osv5m_test --grid $MAPS --fields data/energy/fields.npz \
          --tag pilot_cont_b_maps ;;
    pilot_check) run_cpu pilot_check $PY nc/pilot_check.py ;;
    stage_e_s*)
      seed=${job#stage_e_s}
      run_on_gpu $gpu $job energy.refine --coarse saved_models/energy/cell_b_maps_s$seed.pt \
          --cache data/energy/cache --grid $MAPS --run-name $job --seed $seed \
          --resume saved_models/energy/${job}_last.pt WBFLAGS ;;
    bench_encode_*)
      # encode each benchmark once; concurrent first calls would race on its cache
      local b=${job#bench_encode_}
      run_cpu $job env HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu $PY -c "
from energy.benchmark import load_benchmark, encode_benchmark
paths, _ = load_benchmark('$b')
encode_benchmark('$b', paths, 'data/energy/benchmark_cache', device='cuda')" ;;
    bench_*__stage_e_s*)
      local b=${job#bench_}; b=${b%%__*}; seed=${job##*_s}
      run_on_gpu $gpu $job energy.benchmark --coarse saved_models/energy/cell_b_maps_s$seed.pt \
          --refiner saved_models/energy/stage_e_s$seed.pt --benchmark $b \
          --grid $MAPS --tag stage_e_s$seed ;;
    bench_*)
      local b=${job#bench_}; b=${b%%__*}; local run=${job#*__}; arm=${run%_s*}
      run_on_gpu $gpu $job energy.benchmark --coarse saved_models/energy/$run.pt \
          --benchmark $b --grid $(arm_grid $arm) --fields data/energy/fields.npz --tag $run ;;
    *)
      arm=${job%_s*}; seed=${job##*_s}
      run_on_gpu $gpu $job energy.train --run-name $job --seed $seed \
          --resume saved_models/energy/${job}_last.pt --cache $(arm_cache $arm) \
          $(arm_args $arm) $(arm_exclude $arm) --epochs $EPOCHS \
          --fields data/energy/fields.npz WBFLAGS ;;
  esac
}

# --- the scheduler --------------------------------------------------------
state() {  # <job> -> ok | failed | none
  [ -f data/logs/$1.done ] || { echo none; return; }
  grep -qx 0 data/logs/$1.done && echo ok || echo failed
}
worker() {  # <gpu>
  local gpu=$1 job deps d st ready left
  while true; do
    while IFS='|' read -r job deps; do
      [ "$(state $job)" = none ] || continue
      [ -d $CLAIMS/$job ] && continue
      ready=1
      for d in $deps; do
        st=$(state $d)
        if [ $st = failed ]; then  # a dep failed: skip, don't wait forever
          if mkdir $CLAIMS/$job 2>/dev/null; then
            echo "skipped: dep $d failed" >> data/logs/$job.out; echo 3 > data/logs/$job.done
          fi
          ready=0; break
        fi
        [ $st = ok ] || { ready=0; break; }
      done
      [ $ready = 1 ] || continue
      mkdir $CLAIMS/$job 2>/dev/null || continue   # another worker got it
      echo "[$(date)] GPU $gpu claims $job"
      run_job $gpu $job
      continue 2                                   # rescan from the top
    done < <(jobs)
    left=$(jobs | cut -d'|' -f1 | while read -r j; do [ -f data/logs/$j.done ] || echo $j; done | wc -l)
    [ $left = 0 ] && { echo "[$(date)] GPU $gpu: queue empty"; return 0; }
    sleep 60
  done
}

# --- CPU phases -----------------------------------------------------------
phase() {
  case $1 in
  dl_osv)
    run_cpu dl_osv $PY osv5m.py all || return 1
    run_cpu bench_osv_csv $PY - <<'PY'
import pandas as pd, json, os
m = pd.read_csv('data/osv5m/metadata_osv5m.csv', dtype={'id': str})
t = m[m['selection'] == 'test'].sample(n=10000, random_state=330)
os.makedirs('data/benchmarks', exist_ok=True)
t[['image', 'lat', 'lng']].to_csv('data/benchmarks/osv5m_test10k.csv', index=False)
path = 'data/benchmarks/benchmarks.json'
b = json.load(open(path)) if os.path.exists(path) else {}
b['osv5m_test'] = {'meta': 'data/benchmarks/osv5m_test10k.csv', 'images': 'data/osv5m/images'}
json.dump(b, open(path, 'w'), indent=2)
print('osv5m_test: 10000 rows')
PY
    ;;
  dl_mp16) run_cpu dl_mp16 $PY mp16.py all ;;
  dl_bench)
    # Im2GPS3k / YFCC4k: metadata from the G3 authors' HF repo, then every image
    # still publicly reachable (Flickr, and the Multimedia Commons YFCC100M
    # mirror): ~82% of each, i.e. SUBSETS -- comparable across our runs, not
    # with published numbers. See nc/fetch_benchmarks.py.
    run_cpu dl_bench_meta $PY -c "
from huggingface_hub import hf_hub_download
for n in ('im2gps3k', 'yfcc4k'):
    print(hf_hub_download('Jia-py/G3-checkpoint', n + '_places365.csv', local_dir='data/benchmarks'))" || return 1
    run_cpu dl_bench $PY nc/fetch_benchmarks.py --workers 16 ;;
  rasters)
    # WorldClim is fetched on the node (10 GB bundle, two layers kept) rather
    # than pushed from a laptop; wait for it if the copy is not there yet
    until [ -f data/rasters/worldclim/wc2.1_30s_bio_12.tif ] && [ ! -f data/rasters/worldclim/wc2.1_30s_bio.zip ]; do sleep 30; done
    run_cpu fields $PY -m energy.grid --out data/energy/grid.npz --fields-only \
        --fields-out data/energy/fields.npz \
        --worldclim-tavg data/rasters/worldclim/wc2.1_30s_bio_1.tif \
        --worldclim-prec data/rasters/worldclim/wc2.1_30s_bio_12.tif \
        --elevation data/rasters/elevation/mn30_grd || return 1
    run_cpu maps $PY -m energy.maps --grid data/energy/grid.npz \
        --out-grid $MAPS --fields data/energy/fields.npz || return 1
    wait_done dl_osv
    run_cpu geo_labels $PY nc/build_geo_rasters.py --grid data/energy/grid.npz \
        --out data/energy/grid_geo.npz --mp16-metadata /nonexistent ;;
  overlap)
    # MP-16 and YFCC4k both come from YFCC100M; Im2GPS3k is Flickr too. Every
    # MP-16 arm waits on this. The id pass needs only metadata, so it runs on
    # the FULL benchmark lists; the near-duplicate pass needs pixels, so it
    # runs on the fetched subsets. exclude_ids.txt is the union of both.
    wait_done dl_mp16; wait_done dl_bench
    mkdir -p $OVERLAP_OUT/id $OVERLAP_OUT/embed
    run_cpu overlap_id $PY -m energy.overlap --benchmarks im2gps3k_full yfcc4k_full \
        --train mp16=data/mp16/metadata_mp16.csv:data/mp16/images \
        --out $OVERLAP_OUT/id --skip-embed || return 1
    run_cpu overlap env HIP_VISIBLE_DEVICES=7 CUDA_VISIBLE_DEVICES=7 $PY -m energy.overlap \
        --benchmarks im2gps3k yfcc4k --train mp16=data/mp16/metadata_mp16.csv:data/mp16/images \
        --out $OVERLAP_OUT/embed --radius-km 0.25 || return 1
    # 0.25 km, not the 1 km default: 1.17M MP-16 photos sit within 1 km of a
    # benchmark photo (the cap is 500k); 151k within 0.25 km, still far wider
    # than a re-upload's geotag drifts. The id pass above has no radius.
    # union, plus the mp16_-prefixed twin for the combined cache's id space
    $PY -c "
ids = set()
for f in ('$OVERLAP_OUT/id/exclude_ids.txt', '$OVERLAP_OUT/embed/exclude_ids.txt'):
    ids |= {l.strip() for l in open(f) if l.strip()}
ids = sorted(ids)
open('$OVERLAP_OUT/exclude_ids.txt', 'w').write(''.join(i + '\\n' for i in ids))
open('$OVERLAP_OUT/exclude_ids_combined.txt', 'w').write(''.join('mp16_' + i + '\\n' for i in ids))
print(len(ids), 'excluded ids (id pass: full lists; near-duplicate pass: fetched subsets)')" ;;
  *) echo "unknown phase $1"; return 2 ;;
  esac
}

case $1 in
  launch)
    for p in dl_osv dl_mp16 dl_bench rasters overlap $(seq -f 'gpu%g' 0 $((N - 1))); do
      tmux has-session -t $p 2>/dev/null && { echo "tmux $p exists"; continue; }
      if [[ $p == gpu* ]]; then cmd="worker ${p#gpu}"; else cmd="$p"; fi
      tmux new-session -d -s $p "bash $REPO/nc/cont_pipeline.sh $cmd; echo EXIT_$p rc=\$?; sleep 86400"
      echo "launched $p"
    done ;;
  worker) worker $2 ;;
  status)
    jobs | cut -d'|' -f1 | while read -r j; do
      st=$(state $j); [ $st = none ] && [ -d $CLAIMS/$j ] && st=running
      [ $st = none ] && st=waiting
      printf '%-32s %s\n' $j $st
    done ;;
  unclaim) rmdir $CLAIMS/$2 && echo "released $2" ;;
  *) phase "$@" ;;
esac
