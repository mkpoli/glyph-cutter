"""Read the pages of a book: image files, a directory of them, or a PDF."""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class Page:
    """One page ready to cut."""

    source: Path
    source_id: str
    number: int  # 1-based, within its source
    image: Image.Image


def parse_pages(text: str | None) -> set[int] | None:
    """A page selection such as `1-5,8,11-13` as 1-based page numbers, or None for all."""
    if not text:
        return None
    chosen: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first, last = part.split("-", 1)
            start, end = int(first), int(last)
            if start < 1 or end < start:
                raise ValueError(f"bad page range {part!r}")
            chosen.update(range(start, end + 1))
        else:
            number = int(part)
            if number < 1:
                raise ValueError(f"bad page number {part!r}")
            chosen.add(number)
    return chosen


def _natural_key(path: Path) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _images_in(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=_natural_key,
    )


def _unique(source_id: str, seen: dict[str, int]) -> str:
    count = seen.get(source_id, 0) + 1
    seen[source_id] = count
    return source_id if count == 1 else f"{source_id}-{count}"


def iter_pages(
    inputs: Sequence[str | Path],
    *,
    dpi: int = 300,
    pages: set[int] | None = None,
    max_pages: int | None = None,
) -> Iterator[Page]:
    """Yield the selected pages of every input, in order."""
    seen: dict[str, int] = {}
    emitted = 0
    for given in inputs:
        path = Path(given).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"no such input: {path}")
        if path.is_dir():
            files = _images_in(path)
            if not files:
                raise ValueError(f"no images in {path}")
            for number, file in enumerate(files, start=1):
                if pages is not None and number not in pages:
                    continue
                with Image.open(file) as handle:
                    image = handle.convert("RGB")
                yield Page(file, _unique(file.stem, seen), number, image)
                emitted += 1
                if max_pages is not None and emitted >= max_pages:
                    return
            continue
        if path.suffix.lower() == ".pdf":
            import pymupdf

            with pymupdf.open(path) as document:
                for number, document_page in enumerate(document, start=1):
                    if pages is not None and number not in pages:
                        continue
                    pixmap = document_page.get_pixmap(
                        matrix=pymupdf.Matrix(dpi / 72, dpi / 72), alpha=False
                    )
                    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                    yield Page(path, _unique(path.stem, seen), number, image)
                    emitted += 1
                    if max_pages is not None and emitted >= max_pages:
                        return
            continue
        if path.suffix.lower() in IMAGE_SUFFIXES:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
            yield Page(path, _unique(path.stem, seen), 1, image)
            emitted += 1
            if max_pages is not None and emitted >= max_pages:
                return
            continue
        raise ValueError(f"not an image, a directory of images, or a PDF: {path}")

