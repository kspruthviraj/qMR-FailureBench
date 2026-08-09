#!/usr/bin/env bash
# Reproduce the strict grouped protocol and verify the release artifacts.
#
# Default mode is "verify": it runs integrity tests, creates the versioned MRF
# file only when absent, builds its grouped manifest, and runs the five-seed
# calibration/test protocol only when its frozen output is absent.
#
# Use "legacy" to reproduce the historical baseline pipeline. Use "all" to run
# both modes. No mode performs any network push or repository mutation.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -v PYTHONPATH ]]; then
  export PYTHONPATH="$ROOT:$PYTHONPATH"
else
  export PYTHONPATH="$ROOT"
fi

MODE="verify"
if (( $# > 0 )); then MODE="$1"; fi
if [[ -v V3_FILE ]]; then :; else V3_FILE="data/synthetic/failure_forecast_mrf_v3.h5"; fi
if [[ -v V3_MANIFEST ]]; then :; else V3_MANIFEST="frozen_results/grouped_split_manifest_v3.json"; fi
if [[ -v V3_RESULTS ]]; then :; else V3_RESULTS="frozen_results/grouped_protocol_v3_results.json"; fi
if [[ -v V3_EPOCHS ]]; then :; else V3_EPOCHS="30"; fi
if [[ -v V3_WORKERS ]]; then :; else V3_WORKERS="16"; fi
if [[ -v V3_SEEDS ]]; then :; else V3_SEEDS="42 43 44 45 46"; fi

run_verify() {
  echo "[verify] pytest"
  pytest -q

  echo "[verify] scientific integrity audit"
  python scripts/audit_scientific_integrity.py \
    --output frozen_results/scientific_integrity_audit.json

  if [[ ! -f "$V3_FILE" ]]; then
    echo "[verify] generating versioned MRF data"
    python scripts/generate_v3_data.py \
      --modality mrf --n-signals 50000 --workers "$V3_WORKERS" \
      --mrf-output "$V3_FILE"
  else
    echo "[verify] using existing $V3_FILE"
  fi

  if [[ ! -f "$V3_MANIFEST" ]]; then
    echo "[verify] building grouped split manifest"
    python -m qMR_Robust.data.splits "$V3_FILE" "$V3_MANIFEST" --seed 42
  else
    echo "[verify] using existing $V3_MANIFEST"
  fi

  if [[ ! -f "$V3_RESULTS" ]]; then
    echo "[verify] running grouped protocol"
    python scripts/run_grouped_protocol.py \
      --h5 "$V3_FILE" --manifest "$V3_MANIFEST" \
      --epochs "$V3_EPOCHS" --seeds $V3_SEEDS \
      --output "$V3_RESULTS"
  else
    echo "[verify] using existing $V3_RESULTS"
  fi

  if [[ -f scripts/check_release_consistency.py ]]; then
    echo "[verify] release consistency audit"
    python scripts/check_release_consistency.py
  fi
}

run_legacy() {
  echo "[legacy] data generation and base training"
  python scripts/run_experiment.py
  echo "[legacy] main experiments"
  python scripts/run_v3_final.py
  echo "[legacy] attribution"
  python scripts/run_novel_experiments.py
  echo "[legacy] final analysis"
  python scripts/run_final_analysis.py
  echo "[legacy] sim-to-real checks"
  python scripts/run_sim_to_real_gap.py
  python scripts/run_adaptation_curve.py
  echo "[legacy] seed study"
  python scripts/run_seed_study.py
  echo "[legacy] figures"
  python scripts/generate_key_figures.py
  python scripts/generate_missing_figures.py
}

case "$MODE" in
  verify) run_verify ;;
  legacy) run_legacy ;;
  all) run_legacy; run_verify ;;
  *)
    echo "Usage: bash scripts/reproduce_all.sh [verify|legacy|all]" >&2
    exit 2
    ;;
esac

echo "Reproduction mode '$MODE' completed at $(date)"
