"""CLI entry point for surrogate-null bootstrap calibration."""

from __future__ import annotations

from markovianity_diagnostic.experiments.bootstrap_runner import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
