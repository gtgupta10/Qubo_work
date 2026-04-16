#!/bin/bash
# run_mace_seq.sh — run sequential MACE relaxation (no MPI)
#
# Usage:
#   chmod +x run_mace_seq.sh
#   ./run_mace_seq.sh
# ─────────────────────────────────────────────────────────────────────────────

LOG="mace_seq.log"
PIDFILE="mace_seq.pid"
SCRIPT="mace_seq.py"

# ── Thread control ─────────────────────────────────────────────────────────
# Single process — give it all 16 cores
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TORCH_NUM_THREADS=4

# ── Activate conda env if needed ──────────────────────────────────────────
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate matter

# ── Sanity check ──────────────────────────────────────────────────────────
python -c "from mace.calculators import mace_mp" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: mace not found. Install with:"
    echo "  pip install mace-torch"
    exit 1
fi

echo "Starting sequential MACE job: $(date)"
echo "  Script  : $SCRIPT"
echo "  Log     : $LOG"
echo "  Threads : $TORCH_NUM_THREADS (all 16 cores, no MPI overhead)"

nohup python $SCRIPT > $LOG 2>&1 &

PID=$!
echo $PID > $PIDFILE

echo "  PID     : $PID  (saved to $PIDFILE)"
echo "  Monitor : tail -f $LOG"
echo "  Kill    : kill \$(cat $PIDFILE)"
echo ""
echo "Job running in background. Safe to close terminal."
