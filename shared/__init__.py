#!/usr/bin/env python3
"""shared — the cross-agent contract. Import the handoff types and interfaces from here only."""
from .contract import (Cleaner, CleanRequest, CleanResult, DetectionResult, DetectorAPI,
                       Image, Mask, PipelineOutcome, QAReport, Status, ValidatorAPI, WatermarkType)

__all__ = [
    "Cleaner", "CleanRequest", "CleanResult", "DetectionResult", "DetectorAPI",
    "Image", "Mask", "PipelineOutcome", "QAReport", "Status", "ValidatorAPI", "WatermarkType",
]
