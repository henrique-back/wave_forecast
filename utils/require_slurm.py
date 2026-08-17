"""Guard against launching a GPU-heavy training/search script outside Slurm.

wavetank's GPU is shared across the group via Slurm (see slurm/ and
CLAUDE.md's Commands section) — a job started by directly invoking
`python scripts/optimize.py` / `scripts/train.py` / `scripts/ablate_loss.py`
from a plain shell holds the GPU outside Slurm's queue entirely, invisible
to `squeue`, defeating the group's shared-GPU policy (the exact problem
Slurm's `--gres=gpu:1`-per-job queueing exists to prevent).
"""
import os
import sys


def require_slurm(script_name):
    """Exit immediately unless running inside a Slurm job allocation.

    Checks for SLURM_JOB_ID — set by slurmd for every job step regardless
    of partition/qos/node, so this doesn't need to know anything about
    wavetank/netuno's specific configuration.

    Escape hatch: set WAVE_FORECAST_ALLOW_NO_SLURM=1 for a deliberate
    direct run (a short CPU-only smoke test, local debugging with a tiny
    n_trials/num_epochs, etc.) — prints a warning and continues instead of
    exiting.
    """
    if "SLURM_JOB_ID" in os.environ:
        return
    if os.environ.get("WAVE_FORECAST_ALLOW_NO_SLURM") == "1":
        print(f"[require_slurm] WAVE_FORECAST_ALLOW_NO_SLURM=1 set — running "
              f"{script_name} outside Slurm anyway.", file=sys.stderr)
        return
    sys.exit(
        f"{script_name} must be submitted via Slurm, not run directly — "
        f"wavetank's GPU is shared across the group.\n"
        f"  Copy slurm/run.slurm.template (see slurm/ablate_*.slurm for worked "
        f"examples), then:\n"
        f"    mkdir -p logs\n"
        f"    sbatch slurm/your_job.slurm\n"
        f"  Monitor with: squeue -u $USER\n"
        f"To bypass deliberately (e.g. a short CPU-only smoke test), set "
        f"WAVE_FORECAST_ALLOW_NO_SLURM=1."
    )
