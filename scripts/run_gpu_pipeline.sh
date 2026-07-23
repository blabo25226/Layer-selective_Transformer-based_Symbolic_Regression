#!/usr/bin/env bash
set -euo pipefail

export MPLBACKEND=Agg
cd "$(dirname "$0")/.."

read -r -a PY_CMD <<< "${PY:-python}"
SEEDS=${SEEDS:-"0 1 2 3 4"}
NPS=${NPS:-24}
EVAL_LIMIT=${EVAL_LIMIT:-0}
EPOCHS=${EPOCHS:-8}
LR_GRID=${LR_GRID:-"1e-5 3e-5 1e-4"}
EPOCH_GRID=${EPOCH_GRID:-"4 8"}
PATIENCE=${PATIENCE:-2}
BEAM=${BEAM:-5}
BFGS_RESTARTS=${BFGS_RESTARTS:-5}
BFGS_STOP=${BFGS_STOP:-2.0}
NOISE=${NOISE:-"0.0 0.05 0.1 0.2"}
PYSR=${PYSR:-1}
DREAM4=${DREAM4:-0}
DREAM4_ROOT=${DREAM4_ROOT:-"$PWD/data/dream4"}
DREAM4_SR_TARGETS=${DREAM4_SR_TARGETS:-0}
RANDOM_LAYER_SEEDS=${RANDOM_LAYER_SEEDS:-"0 1 2 3 4"}
NMSE_EQUIV_MARGIN=${NMSE_EQUIV_MARGIN:-0.05}

# --- Speed / robustness knobs (default values reproduce the original behavior) ---
# MAX_PARALLEL_SEEDS>1 runs the per-seed loops of phases 5/6/7/8 concurrently.
# Each seed is an unchanged independent process, so its output is byte-identical
# to running it alone; only wall-clock changes. Keep it small (2-3) on a single
# GPU to stay within VRAM. Thread counts are intentionally left untouched so
# results stay bit-for-bit identical; export OMP_NUM_THREADS yourself if the
# concurrent processes thrash the CPU (that may change BLAS reduction order).
MAX_PARALLEL_SEEDS=${MAX_PARALLEL_SEEDS:-1}
# RESUME=1 re-enters an existing run dir and skips any phase/seed whose final
# output already exists, so a failure late in the pipeline does not force the
# earlier (e.g. 6h Phase 4) phases to recompute.
RESUME=${RESUME:-0}

export LTSR_SEEDS="$SEEDS" LTSR_N_PER_SKELETON="$NPS" LTSR_EVAL_LIMIT="$EVAL_LIMIT"
export LTSR_EPOCHS="$EPOCHS" LTSR_BEAM="$BEAM" LTSR_BFGS_RESTARTS="$BFGS_RESTARTS"
export LTSR_LR_GRID="$LR_GRID" LTSR_EPOCH_GRID="$EPOCH_GRID" LTSR_PATIENCE="$PATIENCE"
export LTSR_BFGS_STOP="$BFGS_STOP" LTSR_NOISE="$NOISE" LTSR_PYSR="$PYSR"
export LTSR_DREAM4="$DREAM4" LTSR_DREAM4_ROOT="$DREAM4_ROOT"
export LTSR_RANDOM_LAYER_SEEDS="$RANDOM_LAYER_SEEDS"
export LTSR_NMSE_EQUIV_MARGIN="$NMSE_EQUIV_MARGIN"

: "${LTSR_WEIGHTS:?Set LTSR_WEIGHTS to the GPU checkpoint path}"
export LTSR_CONFIG=${LTSR_CONFIG:-"$PWD/NSRS/jupyter/100M/config.yaml"}
export LTSR_EQ_SETTING=${LTSR_EQ_SETTING:-"$PWD/NSRS/jupyter/100M/eq_setting.json"}
RUN_ID=${RUN_ID:-"$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short HEAD)"}
export LTSR_RUN_DIR=${LTSR_RUN_DIR:-"$PWD/results/runs/$RUN_ID"}
export LTSR_GRAPH_DIR=${LTSR_GRAPH_DIR:-"$PWD/graphs/$RUN_ID"}
export LTSR_DREAMLIKE_DATA=${LTSR_DREAMLIKE_DATA:-"$LTSR_RUN_DIR/input_data/phase7_dreamlike_v1"}
DATA="$LTSR_RUN_DIR/input_data/diverse_gpu"
CONTRIB="$LTSR_RUN_DIR/phase4_multiseed/layer_ranking_scores.json"

# --- Concurrency helper: run <fn> for each seed with bounded parallelism -------
# Default MAX_PARALLEL_SEEDS=1 => strictly sequential, identical to before.
# Fail-fast: if any seed fails, the run exits non-zero (via set -e on return).
_flush_seed_group() {
  local i
  for i in "${!_pids[@]}"; do
    if ! wait "${_pids[$i]}"; then
      echo "ERROR: seed ${_sds[$i]} failed" >&2
      _rc=1
    fi
  done
  _pids=()
  _sds=()
}

parallel_seeds() {
  local fn="$1" seed
  if [ "${MAX_PARALLEL_SEEDS}" -le 1 ]; then
    for seed in $SEEDS; do "$fn" "$seed"; done
    return 0
  fi
  local -a _pids=() _sds=()
  local _rc=0
  for seed in $SEEDS; do
    "$fn" "$seed" &
    _pids+=("$!")
    _sds+=("$seed")
    if [ "${#_pids[@]}" -ge "${MAX_PARALLEL_SEEDS}" ]; then
      _flush_seed_group
    fi
  done
  _flush_seed_group
  return "$_rc"
}

# _resume_skip <sentinel-file> <human label>: true (skip) iff RESUME=1 and file exists.
_resume_skip() {
  if [ "$RESUME" = "1" ] && [ -f "$1" ]; then
    echo "[resume] $2: found $(basename "$1"), skipping"
    return 0
  fi
  return 1
}

"${PY_CMD[@]}" scripts/preflight_gpu.py \
  --weights "$LTSR_WEIGHTS" --config "$LTSR_CONFIG" --eq-setting "$LTSR_EQ_SETTING"
if [ -e "$LTSR_RUN_DIR" ] || [ -e "$LTSR_GRAPH_DIR" ]; then
  if [ "$RESUME" = "1" ]; then
    echo "[resume] Reusing existing run dir: $LTSR_RUN_DIR"
  else
    echo "ERROR: RUN_ID already exists; choose a new RUN_ID to avoid mixing runs: $RUN_ID" >&2
    exit 2
  fi
fi
mkdir -p "$LTSR_RUN_DIR/logs"
mkdir -p "$LTSR_GRAPH_DIR/figures" "$LTSR_GRAPH_DIR/tables"
exec > >(tee -a "$LTSR_RUN_DIR/logs/pipeline.log") 2>&1
if [ "$RESUME" = "1" ] && [ -f "$LTSR_RUN_DIR/manifest.json" ]; then
  echo "[resume] Existing manifest found; preserving original provenance and appending a resume note."
  "${PY_CMD[@]}" scripts/run_manifest.py resume --run-dir "$LTSR_RUN_DIR" || true
else
  "${PY_CMD[@]}" scripts/run_manifest.py start --run-dir "$LTSR_RUN_DIR" \
    --weights "$LTSR_WEIGHTS" --command "$0 $*" \
    --data-path "$PWD/data/human/gse112372_lps" --data-path "$DREAM4_ROOT"
fi

finish_manifest() {
  status=failed
  if [ "$1" -eq 0 ]; then status=complete; fi
  "${PY_CMD[@]}" scripts/run_manifest.py finish --run-dir "$LTSR_RUN_DIR" --status "$status" || true
}
trap 'finish_manifest $?' EXIT

echo "Run directory: $LTSR_RUN_DIR"
echo "Seeds=$SEEDS n_per_skeleton=$NPS epochs=$EPOCHS beam=$BEAM"
echo "Validation tuning: lr_grid=$LR_GRID epoch_grid=$EPOCH_GRID patience=$PATIENCE"
echo "max_parallel_seeds=$MAX_PARALLEL_SEEDS resume=$RESUME"

if [ "$RESUME" = "1" ] && [ -d "$DATA" ]; then
  echo "[resume] data suite exists, skipping generate_diverse_suite: $DATA"
else
  "${PY_CMD[@]}" scripts/generate_diverse_suite.py --n-per-skeleton "$NPS" \
    --noise $NOISE --tag diverse_gpu --out-root "$LTSR_RUN_DIR/input_data"
fi

# Phase 4 layer selection uses validation only; test remains untouched.
if _resume_skip "$CONTRIB" "Phase4"; then
  :
else
  "${PY_CMD[@]}" scripts/phase4_multiseed.py --data-dir "$DATA" --seeds $SEEDS \
    --epochs "$EPOCHS" --eval-limit "$EVAL_LIMIT" --beam-size "$BEAM" \
    --lr-grid $LR_GRID --epoch-grid $EPOCH_GRID --patience "$PATIENCE" \
    --bfgs-restarts "$BFGS_RESTARTS" --bfgs-stop-time "$BFGS_STOP"
fi

export LTSR_PHASE4_CONTRIB="$CONTRIB"
export LTSR_REQUIRE_LIVE_PHASE4=1

run_phase5_seed() {
  local seed="$1"
  _resume_skip "$LTSR_RUN_DIR/phase5_seed${seed}/selective_results.json" "Phase5 seed${seed}" && return 0
  LTSR_PHASE_TAG="seed${seed}" "${PY_CMD[@]}" scripts/phase5_selective_train.py \
    --data-dir "$DATA" --contributions "$CONTRIB" --seed "$seed" \
    --epochs "$EPOCHS" --eval-limit "$EVAL_LIMIT" --beam-size "$BEAM" \
    --lr-grid $LR_GRID --epoch-grid $EPOCH_GRID --patience "$PATIENCE" \
    --random-layer-seeds $RANDOM_LAYER_SEEDS \
    --bfgs-restarts "$BFGS_RESTARTS" --bfgs-stop-time "$BFGS_STOP"
}
parallel_seeds run_phase5_seed
"${PY_CMD[@]}" scripts/aggregate_phase5_runs.py --run-dir "$LTSR_RUN_DIR" \
  --seeds $SEEDS --k 3 --nmse-equivalence-margin "$NMSE_EQUIV_MARGIN"

run_phase6_seed() {
  local seed="$1"
  _resume_skip "$LTSR_RUN_DIR/phase6_noise_seed${seed}/noise_sweep.json" "Phase6 seed${seed}" && return 0
  LTSR_PHASE_TAG="seed${seed}" "${PY_CMD[@]}" scripts/phase6_noise_sweep.py \
    --noise $NOISE --data-root "$LTSR_RUN_DIR/input_data" --tag diverse_gpu \
    --contributions "$CONTRIB" --seed "$seed" --epochs "$EPOCHS" \
    --eval-limit "$EVAL_LIMIT" --beam-size "$BEAM" \
    --bfgs-restarts "$BFGS_RESTARTS" --bfgs-stop-time "$BFGS_STOP" \
    --rollout 8 --horizon 30 --width 3
}
parallel_seeds run_phase6_seed
"${PY_CMD[@]}" scripts/aggregate_phase6_runs.py --run-dir "$LTSR_RUN_DIR" --seeds $SEEDS

if [ "$DREAM4" = "1" ]; then
  test -d "$DREAM4_ROOT"
  run_phase7_seed() {
    local seed="$1"
    if ! _resume_skip "$LTSR_RUN_DIR/phase7_dream4_size10_seed${seed}/size10_results.json" "Phase7 size10 seed${seed}"; then
      LTSR_PHASE_TAG="seed${seed}" "${PY_CMD[@]}" scripts/phase7_dream4_size10.py \
        --dream4-root "$DREAM4_ROOT" --all-nets --target-limit 0 --seed "$seed" \
        --epochs "$EPOCHS" --beam-size "$BEAM" \
        --bfgs-restarts "$BFGS_RESTARTS" --bfgs-stop-time "$BFGS_STOP"
    fi
    if ! _resume_skip "$LTSR_RUN_DIR/phase7_dream4_size100_seed${seed}/size100_results.json" "Phase7 size100 seed${seed}"; then
      LTSR_PHASE_TAG="seed${seed}" "${PY_CMD[@]}" scripts/phase7_dream4_size100.py \
        --dream4-root "$DREAM4_ROOT" --all-nets --select-all \
        --sr-targets "$DREAM4_SR_TARGETS" --seed "$seed" --epochs "$EPOCHS" \
        --beam-size "$BEAM" --bfgs-restarts "$BFGS_RESTARTS" \
        --bfgs-stop-time "$BFGS_STOP"
    fi
  }
  parallel_seeds run_phase7_seed
  "${PY_CMD[@]}" scripts/aggregate_phase7_runs.py --run-dir "$LTSR_RUN_DIR" --seeds $SEEDS
fi

run_phase8_seed() {
  local seed="$1"
  _resume_skip "$LTSR_RUN_DIR/phase8_lodo_seed${seed}/lodo_results.json" "Phase8 seed${seed}" && return 0
  if [ "$PYSR" = "1" ]; then
    LTSR_PHASE_TAG="seed${seed}" "${PY_CMD[@]}" scripts/phase8_lodo.py \
      --seed "$seed" --epochs "$EPOCHS" --with-pysr --pysr-iters 40
  else
    LTSR_PHASE_TAG="seed${seed}" "${PY_CMD[@]}" scripts/phase8_lodo.py \
      --seed "$seed" --epochs "$EPOCHS"
  fi
}
parallel_seeds run_phase8_seed
"${PY_CMD[@]}" scripts/aggregate_phase8_runs.py --run-dir "$LTSR_RUN_DIR" --seeds $SEEDS

echo "Completed successfully: $LTSR_RUN_DIR"
