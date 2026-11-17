"""Cut a transcribed line into its characters.

The transcription says how many characters a line holds and which; the page says where the ink is.
A line's detections are laid out as *cells* down the line: a detection, the parts of a detection tall
enough to hold two or three characters, and the runs of ink the detector left unboxed between its
boxes. A token of the transcription may take any short run of neighbouring cells, so a box over two
characters is cut, two boxes over one character are joined and a character the detector missed is
found in its ink.

The alignment is the cheapest path through tokens and cells. A placement costs the classifier's
negative log probability of the token over the candidate's crop, and the framing model's negative log
probability that the candidate frames one character at all. A second model accepts a placement for
publication. The character is the transcriber's; the model answers only whether the box frames it.
"""

from __future__ import annotations

import itertools
import json
import math
import statistics
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .ink import GAP, TALL, Frame, columns_of, ink_map, valleys
from .schema import Detection, Line, Placement, Token

FLOOR = -math.log(1e-4)
PROBABILITY_FLOOR = 1e-4
RUN = 3

WEIGHTS = {
    "skip-token": 8.0,
    "skip-detection": 6.0,
    "skip-ink": 1.5,
    "gap": 3.0,
    "frame": 1.0,
    "loose": 2.0,
}

FLAG_SCRIPTS = {"hiragana", "hentaigana", "katakana"}
MARK_KINDS = {"iteration", "voicing", "punctuation", "ligature"}
PUNCTUATION = set("、。〃・，．")


@dataclass
class Cell:
    """One piece of a line a token may take, alone or with its neighbours."""

    along: tuple[int, int]
    across: tuple[int, int]
    source: str  # "detection", "part" or "ink"
    score: float = 0.0
    detection: int | None = None
    parts: int = 1
    column: int = 0
    valley: float = 0.0
    whole: tuple[tuple[int, int], tuple[int, int]] | None = None


@dataclass
class Span:
    """A run of cells as one candidate box."""

    start: int
    end: int
    along: tuple[int, int]
    across: tuple[int, int]
    kind: str
    score: float
    valley: float
    penalty: float = 0.0
    index: int = -1
    cells: int = 1
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class Work:
    """One container of a line with everything the alignment reads."""

    line_id: str
    tokens: list[Token]
    cells: list[Cell]
    spans: list[Span]
    costs: np.ndarray
    known: list[bool]
    frame: Frame
    sizes: tuple[float, float]


def _ink_cells(
    frame: Frame,
    along: tuple[int, int],
    across: tuple[int, int],
    height: float,
    width: float,
    column: int,
) -> list[Cell]:
    """The runs of ink in a stretch of a line no detection covers, each a cell."""
    patch = frame.patch(along, across)
    if patch.size == 0:
        return []
    rows = patch.sum(axis=1)
    top = max(0, along[0])
    runs: list[list[int]] = []
    for row in np.flatnonzero(rows > 0):
        if runs and row - runs[-1][1] <= max(2, int(0.1 * height)):
            runs[-1][1] = int(row) + 1
        else:
            runs.append([int(row), int(row) + 1])
    cells = []
    for start, end in runs:
        if int(rows[start:end].sum()) < 0.02 * height * width or end - start < 3:
            continue
        pieces = [(start, end)]
        if end - start > TALL[0] * height:
            count = min(6, max(2, round((end - start) / height)))
            cuts = [start + cut for cut, _ in valleys(rows[start:end].astype(np.float32), count)]
            edges = [start, *cuts, end]
            pieces = list(itertools.pairwise(edges))
        for first, last in pieces:
            tight = frame.tight((top + first, top + last), across)
            if tight is not None:
                cells.append(Cell(along=tight[0], across=tight[1], source="ink", column=column))
    return cells


def nested(detections: Sequence[Detection]) -> dict[int, list[int]]:
    """Each detection that holds others, with the ones it holds."""
    order = sorted(
        range(len(detections)), key=lambda index: -detections[index].box.w * detections[index].box.h
    )
    holders: dict[int, list[int]] = {}
    held: set[int] = set()
    for position, outer in enumerate(order):
        if outer in held:
            continue
        a = detections[outer].box
        for inner in order[position + 1 :]:
            if inner in held:
                continue
            b = detections[inner].box
            if a.intersection(b) >= 0.7 * b.area:
                holders.setdefault(outer, []).append(inner)
                held.add(inner)
    return holders


def cells_of(
    line: Line,
    detections: Sequence[Detection],
    frame: Frame,
    sizes: tuple[float, float],
) -> tuple[list[Cell], tuple[float, float]]:
    """A line's cells in reading order and the median extent of its characters."""
    if not detections:
        if line.box is None:
            return [], sizes
        along, across = frame.of(line.box)
        if across[1] - across[0] > 2.4 * sizes[1]:
            return [], sizes
        return _ink_cells(frame, along, across, sizes[0], sizes[1], 0), sizes
    holders = nested(detections)
    inside = {inner for held in holders.values() for inner in held}
    kept = [index for index in range(len(detections)) if index not in inside]
    extents = {index: frame.of(detections[index].box) for index in range(len(detections))}
    if len(kept) >= 3:
        height = statistics.median(extents[index][0][1] - extents[index][0][0] for index in kept)
        width = statistics.median(extents[index][1][1] - extents[index][1][0] for index in kept)
    else:
        height, width = sizes
    if line.vertical and len(kept) > 1:
        columns = [
            [kept[position] for position in column]
            for column in columns_of([detections[index].box for index in kept])
        ]
    else:
        columns = [kept]
    columns = [sorted(column, key=lambda index: extents[index][0][0]) for column in columns]
    limits = frame.of(line.box) if line.box is not None else None
    cells: list[Cell] = []
    for number, column in enumerate(columns):
        centre = statistics.median((extents[index][1][0] + extents[index][1][1]) / 2 for index in column)
        half = 0.65 * max(
            width, statistics.median(extents[index][1][1] - extents[index][1][0] for index in column)
        )
        strip = (int(centre - half), int(centre + half))
        cursor = limits[0][0] if limits is not None and len(columns) == 1 else None
        for index in column:
            along, across = extents[index]
            if cursor is not None and along[0] - cursor >= GAP * height:
                cells += _ink_cells(frame, (cursor, along[0]), strip, height, width, number)
            parts = [
                Cell(
                    along=extents[inner][0],
                    across=extents[inner][1],
                    source="part",
                    score=detections[inner].score,
                    detection=index,
                    parts=len(holders[index]),
                    column=number,
                    whole=(along, across),
                )
                for inner in sorted(holders.get(index, ()), key=lambda inner: extents[inner][0][0])
            ]
            if 2 <= len(parts) <= RUN and all(_stacked(a, b) for a, b in itertools.pairwise(parts)):
                cells += parts
            else:
                cells += detection_cells(frame, along, across, detections[index].score, index, number, height)
            cursor = max(cursor or along[1], along[1])
        if (
            limits is not None
            and len(columns) == 1
            and cursor is not None
            and limits[0][1] - cursor >= GAP * height
        ):
            cells += _ink_cells(frame, (cursor, limits[0][1]), strip, height, width, number)
    return cells, (height, width)


def detection_cells(
    frame: Frame, along, across, score: float, index: int, column: int, height: float
) -> list[Cell]:
    """A detection as one cell, or as its parts when it is long enough for two or three characters."""
    tall = (along[1] - along[0]) / max(1.0, height)
    parts = 3 if tall >= TALL[1] else 2 if tall >= TALL[0] else 1
    whole = [
        Cell(along=along, across=across, source="detection", score=score, detection=index, column=column)
    ]
    if parts == 1:
        return whole
    profile = frame.patch(along, across).sum(axis=1).astype(np.float32)
    cuts = valleys(profile, parts)
    if len(cuts) != parts - 1:
        return whole
    top = max(0, along[0])
    edges = [along[0], *[top + cut for cut, _ in cuts], along[1]]
    depth = max(share for _, share in cuts)
    cells = []
    for first, last in itertools.pairwise(edges):
        tight = frame.tight((first, last), across)
        if tight is None:
            return whole
        cells.append(
            Cell(
                along=tight[0],
                across=tight[1],
                source="part",
                score=score,
                detection=index,
                parts=parts,
                column=column,
                valley=depth,
                whole=(along, across),
            )
        )
    return cells


def spans_of(cells: Sequence[Cell], sizes: tuple[float, float]) -> list[Span]:
    """The runs of up to `RUN` neighbouring cells of one column that stack down the line."""
    height = sizes[0]
    spans = []
    for start in range(len(cells)):
        for end in range(start + 1, min(len(cells), start + RUN) + 1):
            run = cells[start:end]
            if run[-1].column != run[0].column or (len(run) > 1 and not _stacked(run[-2], run[-1])):
                break
            along = (min(c.along[0] for c in run), max(c.along[1] for c in run))
            across = (min(c.across[0] for c in run), max(c.across[1] for c in run))
            kind = _kind(cells, start, end)
            if kind == "box" and run[0].whole is not None:
                along, across = run[0].whole
            elif kind != "box" and along[1] - along[0] > 2.8 * height:
                continue
            spans.append(
                Span(
                    start=start,
                    end=end,
                    along=along,
                    across=across,
                    kind=kind,
                    cells=len(run),
                    score=min(c.score for c in run),
                    valley=max(c.valley for c in run),
                )
            )
    for index, span in enumerate(spans):
        span.index = index
    return spans


def _stacked(first: Cell, second: Cell) -> bool:
    """Whether two cells follow each other down the line rather than standing side by side."""
    shared = min(first.across[1], second.across[1]) - max(first.across[0], second.across[0])
    narrow = min(first.across[1] - first.across[0], second.across[1] - second.across[0])
    return shared >= 0.3 * narrow and second.along[0] >= first.along[0] + 0.5 * (
        first.along[1] - first.along[0]
    )


def _kind(cells: Sequence[Cell], start: int, end: int) -> str:
    run = cells[start:end]
    if all(cell.source == "ink" for cell in run):
        return "ink"
    if any(cell.source == "ink" for cell in run):
        return "mixed"
    whole = True
    detections = []
    for cell in run:
        if cell.detection not in detections:
            detections.append(cell.detection)
    for detection in detections:
        inside = sum(1 for cell in run if cell.detection == detection)
        if inside != run[[cell.detection for cell in run].index(detection)].parts:
            whole = False
    if whole:
        return "box" if len(detections) == 1 else "merge"
    return "part" if len(detections) == 1 else "mixed"


def _skips(cells: Sequence[Cell], weights: dict[str, float]) -> list[list[tuple[int, float]]]:
    """From each cell boundary, the ways to leave cells without a token."""
    moves: list[list[tuple[int, float]]] = [[] for _ in range(len(cells) + 1)]
    for index, cell in enumerate(cells):
        moves[index].append(
            (index + 1, weights["skip-ink"] if cell.source == "ink" else weights["skip-detection"])
        )
        if cell.source == "part" and (index == 0 or cells[index - 1].detection != cell.detection):
            last = index
            while last < len(cells) and cells[last].detection == cell.detection:
                last += 1
            if last - index > 1:
                moves[index].append((last, weights["skip-detection"]))
    return moves


def _edges(
    tokens: Sequence[Token],
    spans: Sequence[Span],
    cells: Sequence[Cell],
    costs: np.ndarray,
    weights: dict[str, float],
):
    """Every transition of the lattice as `(token index, from boundary, to boundary, cost, span, places)`."""
    skips = _skips(cells, weights)
    by_start: list[list[Span]] = [[] for _ in range(len(cells) + 1)]
    for span in spans:
        by_start[span.start].append(span)
    edges = []
    for i in range(len(tokens) + 1):
        token = tokens[i] if i < len(tokens) else None
        blank = token is not None and token.text.isspace()
        for c in range(len(cells) + 1):
            if token is not None:
                free = 0.0 if blank else weights["gap"] if token.kind == "gap" else weights["skip-token"]
                edges.append((i, c, i + 1, c, free, None, True))
                if not blank and token.kind != "gap":
                    for span in by_start[c]:
                        edges.append(
                            (i, c, i + 1, span.end, float(costs[i, span.index]) + span.penalty, span, True)
                        )
            for target, cost in skips[c] if c < len(cells) else ():
                edges.append((i, c, i, target, cost, None, False))
    return edges


def _lattice(tokens: Sequence[Token], cells: Sequence[Cell], edges):
    """The cheapest path's token edges, each token's margin, and each chosen edge's posterior."""
    n, m = len(tokens), len(cells)
    infinity = float("inf")
    out: dict[tuple[int, int], list] = {}
    for edge in edges:
        out.setdefault((edge[0], edge[1]), []).append(edge)
    best = np.full((n + 1, m + 1), infinity)
    total = np.full((n + 1, m + 1), infinity)
    back: dict[tuple[int, int], Any] = {}
    best[0, 0] = total[0, 0] = 0.0
    for i in range(n + 1):
        for c in range(m + 1):
            here, mass = best[i, c], total[i, c]
            if here == infinity:
                continue
            for edge in out.get((i, c), ()):
                target = (edge[2], edge[3])
                if here + edge[4] < best[target]:
                    best[target] = here + edge[4]
                    back[target] = edge
                total[target] = -np.logaddexp(-total[target], -(mass + edge[4]))
    after = np.full((n + 1, m + 1), infinity)
    rest = np.full((n + 1, m + 1), infinity)
    after[n, m] = rest[n, m] = 0.0
    for i in range(n, -1, -1):
        for c in range(m, -1, -1):
            for edge in out.get((i, c), ()):
                target = (edge[2], edge[3])
                after[i, c] = min(after[i, c], after[target] + edge[4])
                rest[i, c] = -np.logaddexp(-rest[i, c], -(rest[target] + edge[4]))
    chosen: list = [None] * n
    state = (n, m)
    while state != (0, 0) and state in back:
        edge = back[state]
        if edge[6]:
            chosen[edge[0]] = edge
        state = (edge[0], edge[1])
    cheapest = best[n, m]
    margins = [infinity] * n
    posteriors = [0.0] * n
    for edge in edges:
        if not edge[6]:
            continue
        i = edge[0]
        through = best[i, edge[1]] + edge[4] + after[edge[2], edge[3]]
        if chosen[i] is edge:
            mass = total[i, edge[1]] + edge[4] + rest[edge[2], edge[3]]
            posteriors[i] = float(math.exp(min(0.0, total[n, m] - mass)))
        elif through - cheapest < margins[i]:
            margins[i] = float(through - cheapest)
    return chosen, margins, posteriors


def describe(
    span: Span, frame: Frame, sizes: tuple[float, float], probabilities: np.ndarray
) -> dict[str, float]:
    """What is known about a candidate box before any token is put on it."""
    along, across = span.along, span.across
    extent, breadth = along[1] - along[0], across[1] - across[0]
    patch = frame.patch(along, across)
    density = float(patch.mean()) if patch.size else 0.0
    before = frame.patch((along[0] - 3, along[0] - 1), across)
    after = frame.patch((along[1] + 1, along[1] + 3), across)
    left = frame.patch(along, (across[0] - 3, across[0] - 1))
    right = frame.patch(along, (across[1] + 1, across[1] + 3))
    top = float(probabilities.max())
    entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum())
    tall = math.log(max(extent, 1) / max(sizes[0], 1.0))
    wide = math.log(max(breadth, 1) / max(sizes[1], 1.0))
    return {
        **{f"kind_{kind}": float(span.kind == kind) for kind in ("box", "part", "merge", "ink", "mixed")},
        "cells": float(span.cells),
        "tall": tall,
        "wide": wide,
        "tall_abs": abs(tall),
        "wide_abs": abs(wide),
        "score": span.score,
        "valley": min(span.valley, 2.0),
        "top": top,
        "entropy": entropy,
        "density": density,
        "ink_before": float(before.any(axis=0).mean()) if before.size else 0.0,
        "ink_after": float(after.any(axis=0).mean()) if after.size else 0.0,
        "ink_beside": max(
            float(left.any(axis=1).mean()) if left.size else 0.0,
            float(right.any(axis=1).mean()) if right.size else 0.0,
        ),
        "quality": 0.0,
        "quality_given": 0.0,
        "shift": 0.0,
    }


class Logistic:
    """A logistic regression read from a JSON file: standardised features, weights, a bias."""

    def __init__(self, path: Path):
        model = json.loads(Path(path).read_text(encoding="utf-8"))
        self.names = model["features"]
        self.mean = np.array(model["mean"])
        self.scale = np.array(model["scale"])
        self.weights = np.array(model["weights"])
        self.bias = float(model["bias"])
        self.accept = float(model.get("accept", 0.5))
        self.version = model.get("version", "")

    def probabilities(self, rows: Sequence[dict[str, float]]) -> np.ndarray:
        if not rows:
            return np.zeros(0)
        values = np.array([[row[name] for name in self.names] for row in rows])
        return 1.0 / (1.0 + np.exp(-(((values - self.mean) / self.scale) @ self.weights + self.bias)))


def page_sizes(detections: Sequence[Detection]) -> tuple[float, float]:
    """The page's median character height and width."""
    if not detections:
        return (48.0, 48.0)
    return (
        statistics.median(d.box.h for d in detections),
        statistics.median(d.box.w for d in detections),
    )


def neutral_quality(model: Logistic | None) -> dict[str, float]:
    """The training means of the refiner's three features, for a run without a refiner.

    Without a refiner the model has no expected IoU to offer. Its features are set to the means they
    were fitted at, which keeps them out of the comparison between candidates.
    """
    if model is None:
        return {}
    index = {name: position for position, name in enumerate(model.names)}
    return {
        name: float(model.mean[index[name]])
        for name in ("quality", "quality_given", "shift")
        if name in index
    }


def read(
    works: list[Work],
    image,
    classifier,
    *,
    refiner: Any = None,
    neutral: dict[str, float] | None = None,
) -> list[Work]:
    """`works` with every candidate box read by the classifier and described."""
    spans = [(work, span) for work in works for span in work.spans]
    boxes = [work.frame.box(span.along, span.across) for work, span in spans]
    quality: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * len(boxes)
    if refiner is not None and boxes:
        refined = refiner.refine(np.asarray(image.convert("L")), boxes)
        for position, ((work, span), given, (box, after, before)) in enumerate(
            zip(spans, boxes, refined, strict=True)
        ):
            kept = box.iou(given)
            if kept >= 0.5:
                span.along, span.across = work.frame.of(box)
                boxes[position] = box
                quality[position] = (after, before, 1.0 - kept)
            else:
                quality[position] = (before, before, 0.0)
    crops = [image.crop((int(box.x), int(box.y), int(box.x2), int(box.y2))) for box in boxes]
    classes = classifier.classes
    index = {name: position for position, name in enumerate(classes)}
    probabilities = classifier.probabilities_many(crops) if crops else np.zeros((0, len(classes)))
    first = 0
    for work in works:
        own_rows = probabilities[first : first + len(work.spans)]
        work.costs = np.full((len(work.tokens), max(1, len(work.spans))), FLOOR)
        for i, token in enumerate(work.tokens):
            positions = [index[point] for point in token.code_points if point in index]
            work.known.append(bool(positions))
            if positions and len(work.spans):
                mass = own_rows[:, positions].sum(axis=1)
                work.costs[i, : len(work.spans)] = -np.log(np.maximum(mass, PROBABILITY_FLOOR))
        for offset, (span, row) in enumerate(zip(work.spans, own_rows, strict=True)):
            span.features = describe(span, work.frame, work.sizes, row)
            after, before, shift = quality[first + offset]
            span.features.update({"quality": after, "quality_given": before, "shift": shift})
            if neutral:
                span.features.update(neutral)
        first += len(work.spans)
    return works


def place(work: Work, weights: dict[str, float], framing: Logistic | None) -> list[Placement]:
    """Place one container's tokens on its cells and describe each placement."""
    tokens, spans = work.tokens, work.spans
    if framing is not None and spans:
        framed = framing.probabilities([span.features for span in spans])
        for span, value in zip(spans, framed, strict=True):
            span.penalty = weights["frame"] * min(8.0, -math.log(max(float(value), 1e-6)))
    else:
        for span in spans:
            span.penalty = 0.0 if span.kind == "box" else weights["loose"]
    chosen, margins, posteriors = _lattice(
        tokens, work.cells, _edges(tokens, spans, work.cells, work.costs, weights)
    )
    placements = []
    for i, token in enumerate(tokens):
        span = chosen[i][5] if chosen[i] is not None else None
        if span is None:
            placements.append(Placement(token=token, kind="gap" if token.kind == "gap" else "skip"))
            continue
        placements.append(
            Placement(
                token=token,
                box=work.frame.box(span.along, span.across),
                kind=span.kind,
                cost=float(work.costs[i, span.index]),
            )
        )
    boxed = [p for p in placements if p.box is not None]
    line_cost = statistics.fmean(p.cost for p in boxed) if boxed else FLOOR
    whole = sum(p.kind == "box" for p in boxed) / max(1, len(placements))
    placed = len(boxed) / max(1, len(placements))
    for i, placement in enumerate(placements):
        span = chosen[i][5] if chosen[i] is not None else None
        if span is None:
            continue
        before = placements[i - 1] if i else None
        after = placements[i + 1] if i + 1 < len(placements) else None
        passed = chosen[i][1] - (chosen[i - 1][3] if i and chosen[i - 1] is not None else 0)
        placement.features = {
            **span.features,
            "cost": placement.cost,
            "leads": float(math.exp(-placement.cost) >= span.features["top"] - 1e-6),
            "known": float(work.known[i]),
            "margin": min(margins[i], 20.0),
            "log_posterior": math.log(max(posteriors[i], 1e-6)),
            "frame": span.penalty,
            "passed": float(max(0, passed)),
            "before": float(before is not None and before.box is not None),
            "after": float(after is not None and after.box is not None),
            "before_cost": before.cost if before is not None and before.box is not None else FLOOR,
            "after_cost": after.cost if after is not None and after.box is not None else FLOOR,
            "between": float(
                before is not None
                and before.kind == "box"
                and after is not None
                and after.kind == "box"
            ),
            "line_cost": line_cost,
            "line_whole": whole,
            "line_placed": placed,
            "line_tokens": math.log(len(placements)),
            "kana": float(placement.token.script in FLAG_SCRIPTS),
            "mark": float(placement.token.kind in MARK_KINDS),
        }
        placement.features["cost_margin"] = placement.cost * min(placement.features["margin"], 10.0)
        placement.features["cost_whole"] = placement.cost * span.features["kind_box"]
    return placements


class Cutter:
    """The models a cut reads: the framing model and the acceptance model."""

    def __init__(
        self,
        directory: Path,
        *,
        weights: dict[str, float] | None = None,
        refiner: Any = None,
    ):
        self.weights = {**WEIGHTS, **(weights or {})}
        self.refiner = refiner
        self.framing = Logistic(directory / "frame.json") if (directory / "frame.json").exists() else None
        self.gate = Logistic(directory / "gate.json") if (directory / "gate.json").exists() else None

    @property
    def version(self) -> str:
        return self.gate.version if self.gate is not None else "ungated"

    def cut(
        self,
        lines: Sequence[Line],
        detections: Sequence[Detection],
        image,
        classifier,
    ) -> dict[str, list[Placement]]:
        """Every line of a page cut into its tokens' placements, by line id."""
        result: dict[str, list[Placement]] = {line.id: [] for line in lines}
        for work in prepare_page(
            lines,
            detections,
            image,
            classifier,
            refiner=self.refiner,
            neutral=neutral_quality(self.framing),
        ):
            placements = place(work, self.weights, self.framing)
            boxed = [placement for placement in placements if placement.box is not None]
            if self.gate is not None and boxed:
                for placement, value in zip(
                    boxed, self.gate.probabilities([p.features for p in boxed]), strict=True
                ):
                    placement.probability = float(value)
                    placement.accepted = placement.probability >= self.gate.accept
            elif self.gate is None:
                for placement in boxed:
                    placement.accepted = True
            result[work.line_id] += placements
        return result


def assign_page(lines: Sequence[Line], detections: Sequence[Detection]) -> dict[str, list[Detection]]:
    """Each line's own detections, by containment first and overlap second."""
    own: dict[str, list[Detection]] = {line.id: [] for line in lines}
    for detection in detections:
        holders = [
            line
            for line in lines
            if line.box is not None and line.box.contains(detection.centre)
        ]
        if holders:
            line = min(holders, key=lambda line: line.box.area)
            own[line.id].append(detection)
            continue
        scored = [
            (line.box.iou(detection.box), line)
            for line in lines
            if line.box is not None
        ]
        if not scored:
            continue
        score, line = max(scored, key=lambda pair: pair[0])
        if score >= 0.1:
            own[line.id].append(detection)
    for line in lines:
        if line.vertical:
            own[line.id].sort(key=lambda detection: (detection.centre[1], detection.centre[0]))
        else:
            own[line.id].sort(key=lambda detection: (detection.centre[0], detection.centre[1]))
    return own


def prepare_page(
    lines: Sequence[Line],
    detections: Sequence[Detection],
    image,
    classifier,
    *,
    refiner: Any = None,
    neutral: dict[str, float] | None = None,
) -> list[Work]:
    """A page's lines as cells, candidate boxes and what the models read in each."""
    ink = ink_map(image)
    height, width = page_sizes(detections)
    own = assign_page(lines, detections)
    works: list[Work] = []
    for line in lines:
        frame = Frame(line.vertical, ink)
        sizes = (height, width) if line.vertical else (width, height)
        cells, line_sizes = cells_of(line, own[line.id], frame, sizes)
        works.append(
            Work(
                line_id=line.id,
                tokens=tokens_of(line.text),
                cells=cells,
                spans=spans_of(cells, line_sizes),
                costs=np.zeros((0, 0)),
                known=[],
                frame=frame,
                sizes=line_sizes,
            )
        )
    return read(works, image, classifier, refiner=refiner, neutral=neutral)


def script_of(char: str) -> str:
    point = ord(char)
    if 0x3041 <= point <= 0x309F:
        return "hiragana"
    if 0x30A1 <= point <= 0x30FF:
        return "katakana"
    if 0x3400 <= point <= 0x9FFF or 0xF900 <= point <= 0xFAFF or point >= 0x20000:
        return "han"
    if char.isalpha():
        return "latin"
    return "unknown"


def token_kind(text: str, script: str) -> str:
    point = ord(text) if len(text) == 1 else None
    if point is not None and point in (0x3005, 0x3031, 0x3032, 0x309D, 0x309E, 0x30FD, 0x30FE):
        return "iteration"
    if point is not None and point in (0x3099, 0x309A, 0x309B, 0x309C):
        return "voicing"
    if point is not None and point in (0x309F, 0x30FF):
        return "ligature"
    if point is not None and (0x3001 <= point <= 0x3003 or point in (0x30FB, 0xFF0C, 0xFF0E)):
        return "punctuation"
    if text in {"ゟ", "ヿ"}:
        return "ligature"
    if script == "unknown" and len(text) > 1:
        return "ligature"
    return "char"


def tokens_of(text: str) -> list[Token]:
    """The tokens of a line in reading order, with the code points each may stand for."""
    text = unicodedata.normalize("NFC", text)
    tokens: list[Token] = []
    for position, char in enumerate(text):
        if char.isspace():
            tokens.append(Token(text="", start=position, end=position + 1, kind="gap"))
            continue
        script = script_of(char)
        kind = token_kind(char, script)
        token = Token(
            text=char,
            start=position,
            end=position + 1,
            code_points=frozenset({char}),
            kind=kind,
            script=script,
        )
        if kind == "voicing" and tokens and tokens[-1].kind != "gap":
            base = tokens[-1]
            tokens[-1] = Token(
                text=base.text + char,
                start=base.start,
                end=position + 1,
                code_points=base.code_points | token.code_points,
                kind="char",
                script=base.script,
            )
            continue
        tokens.append(token)
    return tokens
