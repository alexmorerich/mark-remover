"""integration — the neutral coordinator that assembles the six agents and guards the contract.

Not one of the six agents: it owns pipeline assembly, the serial merge process, end-to-end
verification, and gatekeeping of `shared/contract.py`. See integration/README.md.
"""
from .pipeline import build_default_pipeline

__all__ = ["build_default_pipeline"]
