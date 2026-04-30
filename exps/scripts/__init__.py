"""Convenience exports for experiment-running scripts."""

from exps.scripts.run_cgcs import main as run_cgcs_main
from exps.scripts.run_cs import main as run_cs_main

__all__ = [
    "run_cs_main",
    "run_cgcs_main",
]
