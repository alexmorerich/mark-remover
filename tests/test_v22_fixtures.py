#!/usr/bin/env python3
"""V22 §Patch 10 — the regression fixture lists must reference REAL bench assets,
so the categories stay anchored to images that actually exist in the benchmark."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"
BENCH = ROOT / "bench_assets"

FIXTURES = [
    "v22_visible_band_cases.txt",
    "v22_partial_glyph_residue_cases.txt",
    "v22_cover_on_product_reject_cases.txt",
    "v22_recoverable_reject_cases.txt",
]


def _entries(name):
    lines = (FIX / name).read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def test_fixture_files_exist_and_nonempty():
    for name in FIXTURES:
        assert (FIX / name).exists(), name
        assert _entries(name), f"{name} has no entries"


def test_fixture_entries_are_real_bench_assets():
    available = {p.name for p in BENCH.iterdir()}
    for name in FIXTURES:
        for entry in _entries(name):
            assert entry in available, f"{name}: {entry} not in bench_assets"
