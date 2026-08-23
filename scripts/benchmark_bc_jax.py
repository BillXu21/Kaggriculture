"""Thin entrypoint: python scripts/benchmark_bc_jax.py [args]."""

import sys

from bc_manager_jax.benchmark import main

if __name__ == "__main__":
    sys.exit(main())
