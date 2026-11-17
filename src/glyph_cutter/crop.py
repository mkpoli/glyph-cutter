"""Turn a character box into a square crop and record it."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .schema import Box


def paper_colour(image: Image.Image) -> tuple[int, int, int]:
    """The page's paper colour, as the per-channel median of a small copy."""
    small = image.convert("RGB").resize((min(128, image.width), min(128, image.height)))
    values = np.asarray(small).reshape(-1, 3)
    return tuple(int(value) for value in np.median(values, axis=0))


def square_box(box: Box, width: int, height: int, padding: float = 0.08) -> Box:
    """The smallest square about `box` with `padding` of its side added on each side.

    The square is moved inside the page where it fits, and centred on an edge where it does not.
    """
    if padding < 0:
        raise ValueError(f"padding must not be negative, got {padding}")
    side = max(box.w, box.h) * (1.0 + 2.0 * padding)
    centre_x, centre_y = box.centre
    left = centre_x - side / 2
    top = centre_y - side / 2
    left = min(max(0.0, left), width - side) if side <= width else (width - side) / 2
    top = min(max(0.0, top), height - side) if side <= height else (height - side) / 2
    return Box(left, top, side, side)


def crop_square(
    image: Image.Image,
    box: Box,
    *,
    padding: float = 0.08,
    size: int = 384,
    paper: tuple[int, int, int] | None = None,
) -> tuple[Image.Image, Box]:
    """A square crop about `box`, filled with paper where the square reaches past the page."""
    square = square_box(box, image.width, image.height, padding)
    left, top = math.floor(square.x), math.floor(square.y)
    side = max(1, math.ceil(square.w))
    colour = paper if paper is not None else paper_colour(image)
    canvas = Image.new("RGB", (side, side), colour)
    source = image.convert("RGB")
    src_left, src_top = max(0, left), max(0, top)
    src_right = min(image.width, left + side)
    src_bottom = min(image.height, top + side)
    if src_right > src_left and src_bottom > src_top:
        canvas.paste(
            source.crop((src_left, src_top, src_right, src_bottom)),
            (src_left - left, src_top - top),
        )
    if size > 0 and (side != size):
        canvas = canvas.resize((size, size), Image.Resampling.LANCZOS)
    return canvas, Box(left, top, side, side)


def save_crop(image: Image.Image, path: Path, *, format: str = "png", quality: int = 92) -> None:
    """Write a crop, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    options: dict = {}
    if format.lower() in ("jpeg", "jpg"):
        options = {"quality": quality, "optimize": True, "subsampling": 0}
    elif format.lower() == "webp":
        options = {"quality": quality, "method": 4}
    else:
        options = {"optimize": True}
    image.save(path, format=format.upper() if format.lower() != "jpg" else "JPEG", **options)


def extension(format: str) -> str:
    return {"jpeg": "jpg", "jpg": "jpg", "webp": "webp"}.get(format.lower(), "png")


@dataclass
class Manifest:
    """A JSONL manifest of crops, one row each, and a run record beside it."""

    directory: Path
    rows: int = 0

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._handle = (self.directory / "manifest.jsonl").open("w", encoding="utf-8")

    def add(self, row: dict) -> None:
        self._handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
        self.rows += 1

    def close(self, run: dict) -> None:
        self._handle.close()
        (self.directory / "run.json").write_text(
            json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
