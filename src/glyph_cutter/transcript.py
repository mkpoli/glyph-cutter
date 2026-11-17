"""Read a transcription's line boxes from JSON or JSONL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .schema import Box, Line


class TranscriptError(ValueError):
    """A transcription file that cannot be used."""


@dataclass(frozen=True)
class Record:
    """One transcribed line, before it is matched to a page."""

    source: str | None
    file: str | None
    page: int | None
    box: Box
    text: str
    vertical: bool


def load(path: str | Path) -> list[Record]:
    """Every line record of a `.json` array or a `.jsonl` file."""
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    if not text.strip():
        return []
    if file.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        document = json.loads(text)
        rows = document.get("lines") if isinstance(document, dict) else document
        if not isinstance(rows, list):
            raise TranscriptError(f"{file}: expected a list of lines, or an object with a 'lines' list")
    records = []
    for number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise TranscriptError(f"{file}:{number}: expected an object")
        box = row.get("box")
        if not isinstance(box, list) or len(box) != 4 or not all(
            isinstance(value, (int, float)) for value in box
        ):
            raise TranscriptError(f"{file}:{number}: box must be [x, y, w, h]")
        text = row.get("text")
        if not isinstance(text, str) or not text:
            raise TranscriptError(f"{file}:{number}: text must be a non-empty string")
        page = row.get("page")
        if page is not None and (not isinstance(page, int) or page < 1):
            raise TranscriptError(f"{file}:{number}: page must be a positive integer")
        records.append(
            Record(
                source=row.get("source"),
                file=row.get("file"),
                page=page,
                box=Box(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                text=text,
                vertical=bool(row.get("vertical", True)),
            )
        )
    return records


def for_page(records: list[Record], *, source: Path, source_id: str, page: int) -> list[Line]:
    """The records that belong to one page, in file order.

    A record names its page by file, by source, or by page number alone. A record with no locator
    belongs to no page, so a transcript for several books cannot put its lines on all of them.
    """
    found = []
    for record in records:
        if record.file is not None:
            if Path(record.file).name != source.name and record.file != str(source):
                continue
        elif record.source is not None:
            if record.source not in (source_id, source.name, str(source)):
                continue
        elif record.page is None:
            continue
        if record.page is not None and record.page != page:
            continue
        found.append(record)
    return [
        Line(
            id=f"{source_id}:{page}:{number}",
            text=record.text,
            vertical=record.vertical,
            box=record.box,
            page_id=f"{source_id}:{page}",
            seq=number,
        )
        for number, record in enumerate(found, start=1)
    ]
