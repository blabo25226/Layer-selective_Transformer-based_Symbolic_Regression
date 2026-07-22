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

"${PY_CMD[@]}" scripts/preflight_gpu.py \
  --weights "$LTSR_WEIGHTS" --config "$LTSR_CONFIG" --eq-setting "$LTSR_EQ_SETTING"
if [ -e "$LTSR_RUN_DIR" ] || [ -e "$LTSR_GRAPH_DIR" ]; then
  echo "ERROR: RUN_ID already exists; choose a new RUN_ID to avoid mixing runs: $RUN_ID" >&2
  exit 2
fi
mkdir -p "$LTSR_RUN_DIR/logs"
mkdir -p "$LTSR_GRAPH_DIR/figures" "$LTSR_GRAPH_DIR/tables"
exec > >(tee "$LTSR_RUN_DIR/logs/pipeline.log") 2>&1
"${PY_CMD[@]}" scripts/run_manifest.py start --run-dir "$LTSR_RUN_DIR" \
  --weights "$LTSR_WEIGHTS" --command "$0 $*" \
  --data-path "$PWD/data/human/gse112372_lps" --data-path "$DREAM4_ROOT"

finish_manifest() {
  status=failed
  if [ "$1" -eq 0 ]; then status=complete; fi
  "${PY_CMD[@]}" scripts/run_manifest.py finish --run-dir "$LTSR_RUN_DIR" --status "$status" || true
}
trap 'finish_manifest $?' EXIT

echo "Run directory: $LTSR_RUN_DIR"
echo "Seeds=$SEEDS n_per_skeleton=$NPS epochs=$EPOCHS beam=$BEAM"
echo "Validation tuning: lr_grid=$LR_GRID epoch_grid=$EPOCH_GRID patience=$PATIENCE"

"${PY_CMD[@]}" scripts/generate_diverse_suite.py --n-per-skeleton "$NPS" \
  --noise $NOISE --tag diverse_gpu --out-root "$LTSR_RUN_DIR/input_data"

# Phase 4 layer selection uses validation only; test remains untouched.
"${PY_CMD[@]}" scripts/phase4_multiseed.py --data-dir "$DATA" --seeds $SEEDS \
  --epochs "$EPOCHS" --eval-limit "$EVAL_LIMIT" --beam-size "$BEAM" \
  --lr-grid $LR_GRID --epoch-grid $EPOCH_GRID --patience "$PATIENCE" \
  --bfgs-restarts "$BFGS_RESTARTS" --bfgs-stop-time "$BFGS_STOP"

CONTRIB="$LTSR_RUN_DIR/phase4_multiseed/layer_ranking_scores.json"
export LTSR_PHASE4_CONTRIB="$CONTRIB"
export LTSR_REQUIRE_LIVE_PHASE4=1
for seed in $SEEDS; do
  LTSR_PHASE_TAG="seed${seed}" "${PY_CMD[@]}" scripts/phase5_selective_train.py \
    --data-dir "$DATA" --contributions "$CONTRIB" --seed "$seed" \
    --epochs "$EPOCHS" --eval-limit "$EVAL_LIMIT" --beam-size "$BEAM" \
    --lr-grid $LR_GRID --epoch-grid $EPOCH_GRID --patience "$PATIENCE" \
    --random-layer-seeds $RANDOM_LAYER_SEEDS \
    --bfgs-restarts "$BFGS_RESTARTS" --bfgs-stop-time "$BFGS_STOP"
done
"${PY_CMD[@]}" scripts/aggregate_phase5_runs.py --run-dir "$LTSR_RUN_DIR" \
  --seeds $SEEDS --k 3 --nmse-equivalence-margin "$NMSE_EQUIV_MARGIN"

for seed in $SEEDS; do
  LTSR_PHASE_TAG="seed${seed}" "${PY_CMD[@]}" scripts/phase6_noise_sweep.py \
    --noise $NOISE --data-root "$LTSR_RUN_DIR/input_data" --tag diverse_gpu \
    --contributions "$CONTRIB" --seed "$seed" --epochs "$EPOCHS" \
    --eval-limit "$EVAL_LIMIT" --beam-size "$BEAM" \
    --bfgs-restarts "$BFGS_RESTARTS" --bfgs-stop-time "$BFGS_STOP" \
    --rollout 8 --horizon 30 --width 3
done
"${PY_CMD[@]}" scripts/aggregate_phase6_runs.py --run-dir "$LTSR_RUN_DIR" --seeds $SEEDS

if [ "$DREAM4" = "1" ]; then
  test -d "$DREAM4_ROOT"
  for seed in $SEEDS; do
    LTSR_PHASE_TAG="seed${seed}" "${PY_CMD[@]}" scripts/phase7_dream4_size10.py \
      --dream4-root "$DREAM4_ROOT" --all-nets --target-limit 0 --seed "$seed" \
      --epochs "$EPOCHS" --beam-size "$BEAM" \
      --bfgs-restarts "$BFGS_RESTARTS" --bfgs-stop-time "$BFGS_STOP"
    LTSR_PHASE_TAG="seed${seed}" "${PY_CMD[@]}" scripts/phase7_dream4_size100.py \
      --dream4-root "$DREAM4_ROOT" --all-nets --select-all \
      --sr-targets "$DREAM4_SR_TARGETS" --seed "$seed" --epochs "$EPOCHS" \
      --beam-size "$BEAM" --bfgs-restarts "$BFGS_RESTARTS" \
      --bfgs-stop-time "$BFGS_STOP"
  done
  "${PY_CMD[@]}" scripts/aggregate_phase7_runs.py --run-dir "$LTSR_RUN_DIR" --seeds $SEEDS
fi

for seed in $SEEDS; do
  if [ "$PYSR" = "1" ]; then
    LTSR_PHASE_TAG="seed${seed}" "${PY_CMD[@]}" scripts/phase8_lodo.py \
      --seed "$seed" --epochs "$EPOCHS" --with-pysr --pysr-iters 40
  else
    LTSR_PHASE_TAG="seed${seed}" "${PY_CMD[@]}" scripts/phase8_lodo.py \
      --seed "$seed" --epochs "$EPOCHS"
  fi
done
"${PY_CMD[@]}" scripts/aggregate_phase8_runs.py --run-dir "$LTSR_RUN_DIR" --seeds $SEEDS

echo "Completed successfully: $LTSR_RUN_DIR"
