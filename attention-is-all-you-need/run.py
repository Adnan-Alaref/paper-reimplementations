from pathlib import Path
import subprocess
import sys


# ── Project root ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent


# ── Run Training ───────────────────────────
print("\n── Running Training ──")

subprocess.run(
    [sys.executable, "-m", "training.train"],
    check=True,
    cwd=PROJECT_ROOT,
)


# ── Run Evaluation ──────────────────────────
print("\n── Running Evaluation ──")

subprocess.run(
    [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        "evaluating/evaluation.ipynb",
    ],
    check=True,
    cwd=PROJECT_ROOT,
)

print("\n── Complete Pipeline Finished ──")