# Shared helpers for National Compute launch scripts. Source, don't run.
ROOT=/mnt/nvme/sp
REPO=$ROOT/spherical-pigeon
PY=$ROOT/venv/bin/python
# W&B goes online only once ~/.netrc carries the api.wandb.ai login; otherwise the
# run logs offline (wandb sync later) — resolved at launch time of each command.
wb_flags() { if grep -qs api.wandb.ai ~/.netrc; then echo "--wandb --wandb-mode online --wandb-project spherical-pigeon"; else echo "--wandb --wandb-mode offline --wandb-project spherical-pigeon"; fi; }
WB=WBFLAGS
export HF_HOME=$ROOT/hf_cache
export PYTHONUNBUFFERED=1
# 236 cores, ~20 python processes: without a cap every torch/BLAS pool spawns 236
# threads and the node thrashes (load avg 1000, GPUs idle). 4 per process is plenty.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

# run_on_gpu <gpu> <run_name> <module> <args...>
# Runs one python module pinned to a GPU, log to data/logs/<run>.out, marker
# data/logs/<run>.done (exit code inside). Blocking; chain in a tmux session.
run_on_gpu() {
  local gpu=$1 run=$2 mod=$3; shift 3
  cd $REPO
  if [ -f data/logs/$run.done ] && grep -q "^0$" data/logs/$run.done; then
    echo "$run already done, skip"; return 0
  fi
  echo "[$(date)] START $run on GPU $gpu: python -m $mod $*" | tee -a data/logs/$run.out
  local args=(); local a
  for a in "$@"; do if [ "$a" = WBFLAGS ]; then args+=($(wb_flags)); else args+=("$a"); fi; done
  echo "  wandb: $(wb_flags)" >> data/logs/$run.out
  # HIP_VISIBLE_DEVICES only. Setting ROCR_VISIBLE_DEVICES=$gpu as well restricts
  # the runtime to one device and HIP then indexes into that 1-device set, so
  # every gpu>0 resolved to "no GPU" and torch silently fell back to CPU.
  if ! HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu $PY -c "import torch,sys; sys.exit(0 if torch.cuda.device_count()==1 else 1)"; then
    echo "[$(date)] ABORT $run: GPU $gpu not visible to torch" | tee -a data/logs/$run.out; echo 2 > data/logs/$run.done; return 2
  fi
  HIP_VISIBLE_DEVICES=$gpu CUDA_VISIBLE_DEVICES=$gpu \
    $PY -m $mod "${args[@]}" >> data/logs/$run.out 2>&1
  local rc=$?
  echo "[$(date)] END $run rc=$rc" | tee -a data/logs/$run.out
  echo $rc > data/logs/$run.done
  return $rc
}

# wait_for_glob '<glob>' — block until a file matching the glob exists
wait_for_glob() {
  while ! ls $1 >/dev/null 2>&1; do sleep 60; done
}

# run_cpu <name> <command...> — like run_on_gpu but for I/O-bound steps that
# need no GPU (download/extract/adapt). Logs to data/logs/<name>.out with a
# <name>.done marker; unlike a bare command inside a tmux_chain string, an
# early exit here still lets the tmux pane's shell continue (tmux_chain's own
# trailing "sleep 3600" was being skipped by an inline `exit 1`, which killed
# the whole pane and erased the error along with it the first time this ran).
run_cpu() {
  local run=$1; shift
  cd $REPO
  if [ -f data/logs/$run.done ] && grep -q "^0$" data/logs/$run.done; then
    echo "$run already done, skip"; return 0
  fi
  echo "[$(date)] START $run: $*" | tee -a data/logs/$run.out
  ( "$@" ) >> data/logs/$run.out 2>&1
  local rc=$?
  echo "[$(date)] END $run rc=$rc" | tee -a data/logs/$run.out
  echo $rc > data/logs/$run.done
  return $rc
}

# kill_run <run_name_flag_value> — kill exactly the process whose argv carries
# this --run-name, never a same-module sibling. Broad patterns like
# "energy.finetune_encoder" match every run of that module at once — that
# killed a 2-hour, zero-checkpoint stage_c_wide run while trying to stop an
# unrelated stage_c_mp16_pilot (2026-09-06). Always kill by run-name.
kill_run() {
  pkill -f "^/mnt/nvme/sp/venv/bin/python -m energy\..*--run-name $1( |$)"
  sleep 6
  pkill -9 -f "^/mnt/nvme/sp/venv/bin/python -m energy\..*--run-name $1( |$)" 2>/dev/null
}

# wait_done <run> — block until data/logs/<run>.done records exit code 0
wait_done() {
  until grep -qx 0 $REPO/data/logs/$1.done 2>/dev/null; do sleep 120; done
}

# tmux_chain <session> <bash -c string>
tmux_chain() {
  local name=$1; shift
  tmux has-session -t $name 2>/dev/null && { echo "tmux $name exists"; return; }
  tmux new-session -d -s $name "bash -c 'source $REPO/nc/lib.sh; $*; echo CHAIN_DONE_$name; sleep 3600'"
  echo "launched tmux $name"
}
