"""Cut the columns of a page that has no transcription.

`cut` places the tokens of a transcription on a line's cells. A page without one has the cells
alone: the detector's boxes, the parts of a box long enough for two characters, and the ink between
boxes. Nothing says how many characters a column holds, so the framing model decides: it gives every
candidate box the probability that it frames one character, and a column is cut into the candidates
whose odds multiply highest. A box becomes its parts when the parts frame characters better than the
whole does, two boxes become one when their union does, and unboxed ink becomes a box when it is more
likely a character than not.

A column runs down the page, and its cells are cut along it by the size of the column's characters.
A box the column leaves whole is then read alone, by its own shape, down it or across it, so two
small characters in a column of large ones, or two side by side, are cut apart.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import cut
from .ink import TALL, ink_map
from .schema import Box, Detection, Line, Piece

MARGIN = 0.0
DROP = 30.0
ALIGN = 0.45
SHORTEST = 1


@dataclass
class Column:
    """A run of the page's detections read as one line, down the page or across one box."""

    work: cut.Work
    members: list[int]
    held: dict[int, list[int]]


def columns(boxes: Sequence[Box], size: float) -> list[list[Box]]:
    """`boxes` chained top to bottom into columns: each joins the column whose last box is centred
    nearest it, within `ALIGN` of `size`, or starts a column."""
    chains: list[list[Box]] = []
    for box in sorted(boxes, key=lambda box: box.y):
        near = [
            chain
            for chain in chains
            if abs(_centre(chain[-1]) - _centre(box)) < ALIGN * size
        ]
        if near:
            min(near, key=lambda chain: abs(_centre(chain[-1]) - _centre(box))).append(box)
        else:
            chains.append([box])
    return chains


def _centre(box: Box) -> float:
    return box.x + box.w / 2


def columns_of(detections: Sequence[Detection], image) -> list[Column]:
    """The page's detections as columns of cells, then every long or wide box as a line of its own."""
    if not detections:
        return []
    ink = ink_map(image)
    sizes = cut.page_sizes(detections)
    boxes = [detection.box for detection in detections]
    position = {id(box): index for index, box in enumerate(boxes)}
    size = statistics.median(max(box.w, box.h) for box in boxes)
    found = [
        _column(
            f"column-{number}",
            True,
            [position[id(box)] for box in group],
            detections,
            ink,
            sizes,
        )
        for number, group in enumerate(columns(boxes, size))
    ]
    inside = {index for column in found for held in column.held.values() for index in held}
    for index, box in enumerate(boxes):
        if index in inside:
            continue
        for vertical, extent, breadth in ((True, box.h, box.w), (False, box.w, box.h)):
            if extent >= TALL[0] * breadth:
                alone = _column(f"alone-{index}", vertical, [index], detections, ink, (breadth, breadth))
                if len(alone.work.cells) > 1:
                    found.append(alone)
    return found


def _column(
    name: str,
    vertical: bool,
    members: list[int],
    detections: Sequence[Detection],
    ink: np.ndarray,
    sizes: tuple[float, float],
) -> Column:
    line = Line(id=name, text="", vertical=vertical, box=None)
    frame = cut.Frame(vertical, ink)
    own = [detections[index] for index in members]
    cells, line_sizes = cut.cells_of(line, own, frame, sizes)
    work = cut.Work(
        line_id=name,
        tokens=[],
        cells=cells,
        spans=cut.spans_of(cells, line_sizes),
        costs=np.zeros((0, 0)),
        known=[],
        frame=frame,
        sizes=line_sizes,
    )
    held = {
        members[outer]: [members[inner] for inner in inners]
        for outer, inners in cut.nested(own).items()
    }
    return Column(work=work, members=members, held=held)


def choose(work: cut.Work, framing: cut.Logistic, *, margin: float = MARGIN) -> list[tuple[cut.Span, float]]:
    """The candidates of a column whose odds of framing a character multiply highest, in order."""
    cells, spans = work.cells, work.spans
    if not spans:
        return []
    framed = framing.probabilities([span.features for span in spans])
    starts: list[list[cut.Span]] = [[] for _ in cells]
    for span in spans:
        starts[span.start].append(span)
    best = [0.0] * (len(cells) + 1)
    taken: list[cut.Span | None] = [None] * len(cells)
    for boundary in range(len(cells) - 1, -1, -1):
        best[boundary] = best[boundary + 1] - (0.0 if cells[boundary].source == "ink" else DROP)
        for span in starts[boundary]:
            p = min(max(float(framed[span.index]), 1e-6), 1 - 1e-6)
            if span.kind not in ("box", "part") and p < 0.5:
                continue
            odds = math.log(p / (1 - p)) - (0.0 if span.kind == "box" else margin) + best[span.end]
            if odds > best[boundary]:
                best[boundary], taken[boundary] = odds, span
    chosen = []
    boundary = 0
    while boundary < len(cells):
        span = taken[boundary]
        if span is None:
            boundary += 1
        else:
            chosen.append((span, float(framed[span.index])))
            boundary = span.end
    return chosen


def cut_page(
    detections: Sequence[Detection],
    image,
    classifier,
    framing: cut.Logistic,
    *,
    refiner: Any = None,
    margin: float = MARGIN,
) -> list[Piece]:
    """The character boxes of a page with no transcription."""
    found = columns_of(detections, image)
    cut.read(
        [column.work for column in found],
        image,
        classifier,
        refiner=refiner,
        neutral=cut.neutral_quality(framing),
    )
    return pieces_of(found, framing, margin=margin)


def pieces_of(found: Sequence[Column], framing: cut.Logistic, *, margin: float = MARGIN) -> list[Piece]:
    """The boxes chosen in columns the models have read."""
    alone = {
        column.members[0]: column for column in found if column.work.line_id.startswith("alone-")
    }
    pieces = []
    for column in found:
        if column.work.line_id.startswith("alone-"):
            continue
        for piece in _pieces(column, framing, margin):
            own = alone.get(piece.detections[0]) if piece.kind == "box" and piece.detections else None
            parts = _pieces(own, framing, margin) if own is not None else []
            pieces += parts if len(parts) > 1 else [piece]
    return pieces


def _pieces(column: Column, framing: cut.Logistic, margin: float) -> list[Piece]:
    work = column.work
    pieces = []
    for span, probability in choose(work, framing, margin=margin):
        made = []
        for cell in work.cells[span.start : span.end]:
            if cell.detection is not None and column.members[cell.detection] not in made:
                member = column.members[cell.detection]
                made += [member, *column.held.get(member, ())]
        pieces.append(
            Piece(
                box=work.frame.box(span.along, span.across),
                kind=span.kind,
                probability=probability,
                detections=tuple(made),
            )
        )
    return pieces


def ordered(pieces: Sequence[Piece], reading: str = "rtl") -> list[Piece]:
    """Pieces in reading order.

    `rtl` reads columns right to left and each column top to bottom, the order of a classical
    Chinese, Japanese or Korean book. `ltr` reads a horizontal page left to right and each line top
    to bottom. `none` leaves the detection order alone, which is useful for debugging.
    """
    if reading == "none" or not pieces:
        return list(pieces)
    width = statistics.median(max(piece.box.w, piece.box.h) for piece in pieces)
    if reading == "ltr":
        return sorted(pieces, key=lambda piece: (piece.box.y, piece.box.x))
    # Group by x, right column first; a column may drift a little without splitting in two.
    columns: list[list[Piece]] = []
    for piece in sorted(pieces, key=lambda piece: -piece.box.centre[0]):
        if columns and abs(columns[-1][-1].box.centre[0] - piece.box.centre[0]) <= 0.5 * width:
            columns[-1].append(piece)
        else:
            columns.append([piece])
    return [piece for column in columns for piece in sorted(column, key=lambda piece: piece.box.y)]
