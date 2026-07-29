#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
: "${LTSR_WEIGHTS:?Set LTSR_WEIGHTS before starting the campaign}"
: "${LTSR_CONFIG:?Set LTSR_CONFIG before starting the campaign}"
: "${LTSR_EQ_SETTING:?Set LTSR_EQ_SETTING before starting the campaign}"
if [ -n "$(git status --porcelain)" ]; then
  echo "ERROR: the working tree must be completely clean, including untracked files." >&2
  echo "       Leftovers from an earlier run block the campaign; commit or remove them:" >&2
  git status --short >&2
  exit 2
fi

CAMPAIGN_ID=${CAMPAIGN_ID:-"$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short HEAD)"}
SMOKE_RUN_ID="${CAMPAIGN_ID}_smoke"
FULL_RUN_ID="${CAMPAIGN_ID}_full"
FULL_RUN_DIR="results/runs/$FULL_RUN_ID"
RUN_SMOKE=${RUN_SMOKE:-1}
PUBLISH_GIT=${PUBLISH_GIT:-0}
ARCHIVE_DIR=${ARCHIVE_DIR:-"$PWD/results/archives"}
# EVAL_LIMIT=0 evaluates every problem. Override it for a timing pilot only; such a
# run must not be used to pick hyperparameters for the final test results.
EVAL_LIMIT=${EVAL_LIMIT:-0}
CAMPAIGN_LOG=${CAMPAIGN_LOG:-"results/runs/${CAMPAIGN_ID}_campaign.log"}

if [ "$EVAL_LIMIT" != "0" ]; then
  echo "WARNING: EVAL_LIMIT=$EVAL_LIMIT - this campaign is a pilot, not a final run." >&2
fi

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
  EPOCHS=${EPOCHS:-8} EVAL_LIMIT="$EVAL_LIMIT" LR_GRID="${LR_GRID:-1e-5 3e-5 1e-4}" \
  EPOCH_GRID="${EPOCH_GRID:-4 8}" PATIENCE=${PATIENCE:-2} BEAM=${BEAM:-5} \
  BFGS_RESTARTS=${BFGS_RESTARTS:-5} BFGS_STOP=${BFGS_STOP:-2.0} \
  NOISE="${NOISE:-0.0 0.05 0.1 0.2}" PYSR=${PYSR:-1} DREAM4=${DREAM4:-1} \
  RANDOM_LAYER_SEEDS="${RANDOM_LAYER_SEEDS:-0 1 2 3 4}" \
  NMSE_EQUIV_MARGIN=${NMSE_EQUIV_MARGIN:-0.05} \
  bash scripts/run_gpu_pipeline.sh

# validate_gpu_run.py records its own outcome in the manifest.
python scripts/validate_gpu_run.py --run-dir "$FULL_RUN_DIR"

# From here the pipeline and the checks have already succeeded, so any later failure
# (archiving, export, push) must still be visible in the manifest.
mark_publication_failed() {
  if [ "$1" -ne 0 ]; then
    python scripts/run_manifest.py stage --run-dir "$FULL_RUN_DIR" \
      --stage publication --status failed || true
    echo "ERROR: publication stage failed after a complete, validated run." >&2
    echo "       Do NOT re-run the campaign; fix the cause and re-run only the" >&2
    echo "       failed step (archive / export_run_summary.py / git push)." >&2
  fi
}
trap 'mark_publication_failed $?' EXIT

mkdir -p "$ARCHIVE_DIR"
ARCHIVE_PATHS=("$FULL_RUN_DIR" "graphs/$FULL_RUN_ID")
for extra in "results/runs/$SMOKE_RUN_ID" "graphs/$SMOKE_RUN_ID" "$CAMPAIGN_LOG"; do
  if [ -e "$extra" ]; then ARCHIVE_PATHS+=("$extra"); fi
done
# The campaign log is still being written, so the archived copy stops here.
tar -czf "$ARCHIVE_DIR/${FULL_RUN_ID}.tar.gz" "${ARCHIVE_PATHS[@]}"
sha256sum "$ARCHIVE_DIR/${FULL_RUN_ID}.tar.gz" \
  > "$ARCHIVE_DIR/${FULL_RUN_ID}.tar.gz.sha256"
python scripts/export_run_summary.py --run-dir "$FULL_RUN_DIR" \
  --archive "$ARCHIVE_DIR/${FULL_RUN_ID}.tar.gz"

if [ "$PUBLISH_GIT" = "1" ]; then
  git add -- "results/published/$FULL_RUN_ID" "graphs/$FULL_RUN_ID"
  git commit -m "Publish GPU run $FULL_RUN_ID"
  git push origin HEAD
fi

trap - EXIT
echo "Campaign complete: $FULL_RUN_ID"
