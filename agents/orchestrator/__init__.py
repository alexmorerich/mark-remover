"""orchestrator agent package — control loop, routing, retry ownership.

Assembly (wiring all six agents) lives in the neutral `integration/` area, not here, so this
agent does not depend on the other five. Import the pipeline from `integration`.
"""
from .config import PipelineConfig
from .orchestrator import Orchestrator

__all__ = ["Orchestrator", "PipelineConfig"]
