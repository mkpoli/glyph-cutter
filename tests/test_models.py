from __future__ import annotations

import json
from pathlib import Path

import pytest

from glyph_cutter import models
from glyph_cutter.cut import Logistic
from glyph_cutter.sources import parse_pages


def test_parse_pages():
    assert parse_pages("1-3,8") == {1, 2, 3, 8}
    assert parse_pages(None) is None
    with pytest.raises(ValueError):
        parse_pages("0-2")


def test_logistic_reads_and_predicts(tmp_path: Path):
    path = tmp_path / "frame.json"
    path.write_text(
        json.dumps(
            {
                "features": ["a", "b"],
                "mean": [0.0, 0.0],
                "scale": [1.0, 1.0],
                "weights": [2.0, -1.0],
                "bias": 0.0,
                "version": "test",
            }
        ),
        encoding="utf-8",
    )
    model = Logistic(path)
    assert model.version == "test"
    assert model.probabilities([{"a": 0.0, "b": 0.0}])[0] == pytest.approx(0.5)
    assert model.probabilities([{"a": 1.0, "b": 0.0}])[0] > 0.8


def test_missing_lists_absent_files(tmp_path: Path):
    assert set(models.missing(tmp_path)) == set(models.FILES)
