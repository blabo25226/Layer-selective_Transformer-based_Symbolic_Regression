#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
: "${LTSR_WEIGHTS:?Set LTSR_WEIGHTS before starting the campaign}"
: "${LTSR_CONFIG:?Set LTSR_CONFIG before starting the campaign}"
: "${LTSR_EQ_SETTING:?Set LTSR_EQ_SETTING before starting the campaign}"
if [ -n "$(git status --porcelain)" ]; then
  echo "ERROR: commit or remove all non-ignored changes before the GPU campaign" >&2
  exit 2
fi

CAMPAIGN_ID=${CAMPAIGN_ID:-"$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short HEAD)"}
SMOKE_RUN_ID="${CAMPAIGN_ID}_smoke"
FULL_RUN_ID="${CAMPAIGN_ID}_full"
RUN_SMOKE=${RUN_SMOKE:-1}
PUBLISH_GIT=${PUBLISH_GIT:-0}
ARCHIVE_DIR=${ARCHIVE_DIR:-"$PWD/results/archives"}

if [ "$PUBLISH_GIT" = "1" ]; then
  git config user.name >/dev/null || {
    echo "ERROR: configure git user.name before using PUBLISH_GIT=1" >&2
    exit 2
  }
  git config user.email >/dev/null || {
    echo "ERROR: configure git user.email before using PUBLISH_GIT=1" >&2
    exit 2
  }
  git remote get-url origin >/dev/null || {
    echo "ERROR: configure the origin remote before using PUBLISH_GIT=1" >&2
    exit 2
  }
fi

if [ "$RUN_SMOKE" = "1" ]; then
  RUN_ID="$SMOKE_RUN_ID" NPS=2 SEEDS="0 1" EPOCHS=1 EVAL_LIMIT=2 \
    LR_GRID="1e-4" EPOCH_GRID="1" PATIENCE=0 BEAM=1 \
    BFGS_RESTARTS=1 BFGS_STOP=0.2 NOISE="0.0" PYSR=0 DREAM4=0 \
    RANDOM_LAYER_SEEDS="0" NMSE_EQUIV_MARGIN=0.05 \
    bash scripts/run_gpu_pipeline.sh
  python scripts/validate_gpu_run.py --run-dir "results/runs/$SMOKE_RUN_ID"
fi

RUN_ID="$FULL_RUN_ID" SEEDS="${SEEDS:-0 1 2 3 4}" NPS=${NPS:-24} \
  EPOCHS=${EPOCHS:-8} EVAL_LIMIT=0 LR_GRID="${LR_GRID:-1e-5 3e-5 1e-4}" \
  EPOCH_GRID="${EPOCH_GRID:-4 8}" PATIENCE=${PATIENCE:-2} BEAM=${BEAM:-5} \
  BFGS_RESTARTS=${BFGS_RESTARTS:-5} BFGS_STOP=${BFGS_STOP:-2.0} \
  NOISE="${NOISE:-0.0 0.05 0.1 0.2}" PYSR=${PYSR:-1} DREAM4=${DREAM4:-1} \
  bash scripts/run_gpu_pipeline.sh

python scripts/validate_gpu_run.py --run-dir "results/runs/$FULL_RUN_ID"
mkdir -p "$ARCHIVE_DIR"
tar -czf "$ARCHIVE_DIR/${FULL_RUN_ID}.tar.gz" \
  "results/runs/$FULL_RUN_ID" "graphs/$FULL_RUN_ID"
sha256sum "$ARCHIVE_DIR/${FULL_RUN_ID}.tar.gz" \
  > "$ARCHIVE_DIR/${FULL_RUN_ID}.tar.gz.sha256"
python scripts/export_run_summary.py --run-dir "results/runs/$FULL_RUN_ID" \
  --archive "$ARCHIVE_DIR/${FULL_RUN_ID}.tar.gz"

if [ "$PUBLISH_GIT" = "1" ]; then
  git add -- "results/published/$FULL_RUN_ID" "graphs/$FULL_RUN_ID"
  git commit -m "Publish GPU run $FULL_RUN_ID"
  git push origin HEAD
fi

echo "Campaign complete: $FULL_RUN_ID"
