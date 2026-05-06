"""CLI entry point for experiment sweeps."""

from __future__ import annotations

from markovianity_diagnostic.experiments.runner import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
