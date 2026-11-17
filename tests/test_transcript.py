from __future__ import annotations

import json
from pathlib import Path

import pytest

from glyph_cutter import transcript
from glyph_cutter.schema import Box


def test_load_jsonl(tmp_path: Path):
    path = tmp_path / "lines.jsonl"
    path.write_text(
        '{"file":"book.pdf","page":1,"box":[1,2,3,4],"text":"字"}\n'
        '{"source":"book","page":2,"box":[5,6,7,8],"text":"詞","vertical":false}\n',
        encoding="utf-8",
    )
    records = transcript.load(path)
    assert len(records) == 2
    assert records[0].box == Box(1, 2, 3, 4)
    assert records[0].vertical is True
    assert records[1].vertical is False


def test_load_json_array(tmp_path: Path):
    path = tmp_path / "lines.json"
    path.write_text(
        json.dumps({"lines": [{"page": 1, "box": [0, 0, 1, 1], "text": "字"}]}),
        encoding="utf-8",
    )
    assert len(transcript.load(path)) == 1


def test_rejects_a_bad_box(tmp_path: Path):
    path = tmp_path / "lines.jsonl"
    path.write_text('{"page":1,"box":[0,0,1],"text":"字"}\n', encoding="utf-8")
    with pytest.raises(transcript.TranscriptError):
        transcript.load(path)


def test_for_page_matches_by_file_and_page(tmp_path: Path):
    records = [
        transcript.Record(None, "book.pdf", 1, Box(0, 0, 10, 10), "一", True),
        transcript.Record(None, "book.pdf", 2, Box(0, 0, 10, 10), "二", True),
        transcript.Record(None, "other.pdf", 1, Box(0, 0, 10, 10), "三", True),
    ]
    lines = transcript.for_page(records, source=tmp_path / "book.pdf", source_id="book", page=2)
    assert [line.text for line in lines] == ["二"]
    assert lines[0].seq == 1


def test_for_page_ignores_a_record_without_a_locator(tmp_path: Path):
    records = [transcript.Record(None, None, None, Box(0, 0, 10, 10), "一", True)]
    assert transcript.for_page(records, source=tmp_path / "book.pdf", source_id="book", page=1) == []
