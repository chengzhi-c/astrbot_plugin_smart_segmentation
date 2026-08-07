from __future__ import annotations

import json
from pathlib import Path

from astrbot_plugin_smart_segmentation.segmentation import (
    parse_segments_from_model_output,
    split_text_locally,
)

GOLDEN_DIR = Path(__file__).parent / "golden"


def _load_cases() -> list[dict]:
    path = GOLDEN_DIR / "local_split_cases.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_local_split_goldens() -> None:
    for case in _load_cases():
        got = split_text_locally(case["text"], max_segments=case.get("max_segments", 5))
        assert got == case["expected"], case.get("name", case["text"])


def test_model_output_goldens() -> None:
    path = GOLDEN_DIR / "model_output_cases.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    for case in cases:
        got = parse_segments_from_model_output(
            case["raw"],
            max_segments=case.get("max_segments", 5),
        )
        assert got == case["expected"], case.get("name", case["raw"])
