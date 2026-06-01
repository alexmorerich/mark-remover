#!/usr/bin/env python3
"""V19 — known-mark registry (patch plan §1.1).

Borrowed from ``remove-ai-watermarks``: instead of scattering Sunsky detection
and removal logic across ``detector.py``, ``mark_remover.py`` and the patch
files, declare the watermark once as a :class:`KnownSunskyMark` that binds its
location, recovery strategy and the detect / remove callables together. This
gives a single extension point if Sunsky ever changes the watermark's font,
opacity, colour or geometry — register a new variant, nothing else moves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import sunsky_reverse_alpha as _ra


@dataclass(frozen=True)
class KnownSunskyMark:
    key: str
    label: str
    location: str            # e.g. "center-band"
    recovery: str            # human-readable recovery strategy
    detect: Optional[Callable] = None
    remove: Optional[Callable] = None
    aspect_lo: float = 5.0   # width/height range of the text line
    aspect_hi: float = 9.0


def detect_sunsky_mark(image, *, detector_module=None):
    """Detect the Sunsky watermark box on ``image`` (BGR). Returns a mark_box
    dict ``{"x","y","w","h"}`` or ``None``. Thin wrapper over the project
    detector so the registry is the single detection entry point."""
    import cv2
    if detector_module is None:
        import detector as detector_module
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    try:
        dets = detector_module.detect_template_image(
            gray, detector_module.get_templates()[0])
    except Exception:
        return None
    if not dets:
        return None
    return dets[0].get("mark_box")


def remove_sunsky_reverse_alpha(image, mark_box, **kwargs):
    """Reverse-alpha recovery entry used by the registry (patch plan §1.2)."""
    return _ra.repair_sunsky_reverse_alpha(image, mark_box, **kwargs)


SUNSKY_ONLINE = KnownSunskyMark(
    key="sunsky_online",
    label="sunsky-online.com centered watermark",
    location="center-band",
    recovery="reverse-alpha + thin residual cleanup",
    detect=detect_sunsky_mark,
    remove=remove_sunsky_reverse_alpha,
)

REGISTRY: List[KnownSunskyMark] = [SUNSKY_ONLINE]


def get_mark(key: str = "sunsky_online") -> Optional[KnownSunskyMark]:
    for m in REGISTRY:
        if m.key == key:
            return m
    return None
