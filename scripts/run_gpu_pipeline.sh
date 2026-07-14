#!/usr/bin/env bash
# Scaled-up GPU pipeline for the LTSR D-action study.
#
# The CPU pilot (scripts/... with --eval-limit 8, 3 seeds, 10M weights) validated
# the approach. This runs the paper-scale version on a GPU machine:
#   - 100M-equation pretrained checkpoint (better prior; same architecture)
#   - larger structure-split suite (more skeletons × params)
#   - 5 seeds for tight layer-contribution CIs (reviewer note A-1)
#   - higher-quality decode (beam, more BFGS restarts/time) and realistic TPSR
#
# Usage:
#   # 1) get the 100M checkpoint (same config as 10M), e.g.:
#   #    huggingface-cli download TommasoBendinelli/NeuralSymbolicRegressionThatScales \
#   #        100M.ckpt --local-dir NSRS/weights
#   export LTSR_WEIGHTS="$PWD/NSRS/weights/100M.ckpt"   # omit to use bundled 10M
#   bash scripts/run_gpu_pipeline.sh
#
# Env knobs (override inline): SEEDS, NPS (n-per-skeleton), EVAL_LIMIT, EPOCHS,
# BEAM, BFGS_RESTARTS, BFGS_STOP, NOISE, PYSR (1 to include PySR in LODO).
set -u
export MPLBACKEND=Agg
cd "$(dirname "$0")/.."
PY=${PY:-python}                    # set PY="conda run -n <env> python" if needed
DATA=results/synthetic/diverse_gpu

SEEDS=${SEEDS:-"0 1 2 3 4"}
NPS=${NPS:-24}
EVAL_LIMIT=${EVAL_LIMIT:-0}         # 0 = all test problems
EPOCHS=${EPOCHS:-8}
BEAM=${BEAM:-5}
BFGS_RESTARTS=${BFGS_RESTARTS:-5}
BFGS_STOP=${BFGS_STOP:-2.0}
NOISE=${NOISE:-"0.0 0.05 0.1 0.2"}
PYSR=${PYSR:-1}

echo "==== GPU PIPELINE ===="
echo "weights: ${LTSR_WEIGHTS:-<bundled 10M>}"
echo "seeds=$SEEDS nps=$NPS eval_limit=$EVAL_LIMIT epochs=$EPOCHS beam=$BEAM"
$PY -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# 1. Larger structure-split suite (+ noise levels for H3)
$PY scripts/generate_diverse_suite.py --n-per-skeleton "$NPS" --noise $NOISE \
    --tag diverse_gpu --out-root results/synthetic

# 2. Single-seed Phase 4 -> contributions.json (ranking source for Phase 5)
$PY scripts/phase4_layer_contribution.py --data-dir "$DATA" --epochs "$EPOCHS" \
    --eval-limit "$EVAL_LIMIT" --beam-size "$BEAM" \
    --bfgs-restarts "$BFGS_RESTARTS" --bfgs-stop-time "$BFGS_STOP"

# 3. Phase 4 multi-seed (layer contribution with CI)
$PY scripts/phase4_multiseed.py --data-dir "$DATA" --seeds $SEEDS --epochs "$EPOCHS" \
    --eval-limit "$EVAL_LIMIT" --beam-size "$BEAM" \
    --bfgs-restarts "$BFGS_RESTARTS" --bfgs-stop-time "$BFGS_STOP"

# 4. Phase 5 selective FT (dynamic ranking, honest random control)
$PY scripts/phase5_selective_train.py --data-dir "$DATA" --epochs "$EPOCHS" \
    --eval-limit "$EVAL_LIMIT" --beam-size "$BEAM" \
    --bfgs-restarts "$BFGS_RESTARTS" --bfgs-stop-time "$BFGS_STOP"

# 5. Phase 6 noise sweep (H3) with a realistic TPSR budget
$PY scripts/phase6_noise_sweep.py --noise $NOISE --data-root results/synthetic \
    --tag diverse_gpu --epochs "$EPOCHS" --eval-limit "$EVAL_LIMIT" \
    --beam-size "$BEAM" --bfgs-restarts "$BFGS_RESTARTS" --bfgs-stop-time "$BFGS_STOP" \
    --rollout 8 --horizon 30 --width 3

# 6. Phase 8 LODO (donor generalization), with PySR head-to-head if requested
if [ "$PYSR" = "1" ]; then
  $PY scripts/phase8_lodo.py --epochs "$EPOCHS" --with-pysr --pysr-iters 40
else
  $PY scripts/phase8_lodo.py --epochs "$EPOCHS"
fi

echo "==== DONE. Reports in results/phase_results/ ===="
