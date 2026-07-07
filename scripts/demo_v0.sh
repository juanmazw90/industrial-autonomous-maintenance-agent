#!/usr/bin/env bash
# AMIA V0 Demo — Generates synthetic dataset and EDA plots
# Usage: ./scripts/demo_v0.sh [--days 365] [--seed 42]
set -euo pipefail

DAYS=365
SEED=42
while [[ $# -gt 0 ]]; do
  case "$1" in
    --days) DAYS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *) echo "Uso: $0 [--days N] [--seed N]"; exit 1 ;;
  esac
done
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "============================================"
echo " AMIA V0 — Synthetic Data Demo"
echo "============================================"
echo " Days: $DAYS  |  Seed: $SEED"
echo " Repo: $REPO_ROOT"
echo ""

# Check uv is installed
if ! command -v uv &>/dev/null; then
  echo "ERROR: 'uv' is required. Install it with: curl -Lsf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

cd "$REPO_ROOT"

echo "[1/3] Generating synthetic industrial dataset..."
uv run scripts/generate_data.py --days "$DAYS" --seed "$SEED" --output data/synthetic

echo ""
echo "[2/3] Generating EDA visualizations..."
uv run scripts/visualize_data.py --input data/synthetic --output data/synthetic/plots

echo ""
echo "[3/3] Summary"
echo "----------------------------------------------"
if command -v python3 &>/dev/null; then
  python3 -c "
import json, pathlib
stats = json.loads(pathlib.Path('data/synthetic/dataset_stats.json').read_text())
print(f'  Rows generated:   {stats[\"total_rows\"]:,}')
print(f'  Machines:         {stats[\"machines\"]}')
print(f'  Failure events:   {stats[\"failure_events\"]}')
print(f'  Rows with RUL:    {stats[\"rows_with_rul\"]:,}')
print()
print('  Failure mode distribution:')
for mode, count in sorted(stats['failure_mode_counts'].items(), key=lambda x: -x[1]):
    bar = '█' * (count // 500)
    print(f'    {mode:<22} {count:>6,}  {bar}')
"
fi
echo ""
echo "Plots saved to: data/synthetic/plots/"
echo "Dataset:        data/synthetic/sensor_readings.{parquet,csv}"
echo ""
echo "Next step → V1: RAG + Chat Agent"
echo "  See the roadmap in .claude/plans/ for details."
