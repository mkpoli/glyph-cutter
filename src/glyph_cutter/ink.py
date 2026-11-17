"""The ink of a page and the frame a line's cells are laid out in."""

from __future__ import annotations

import itertools
import statistics
from collections.abc import Sequence

import numpy as np

from .schema import Box

#: A detection this many median heights tall is offered in two parts, and in three.
TALL = (1.35, 2.3)
#: A gap between detections this share of a median height is searched for ink.
GAP = 0.45
#: Column grouping thresholds, measured on a page of 蝦夷紀行 with ten visible lines.
GAP_RATIO = 0.7
MERGE_RATIO = 1.6


def ink_map(image) -> np.ndarray:
    """Which pixels of a page are ink: darker than the paper around them by a margin the page sets."""
    import cv2

    grey = np.asarray(image.convert("L"))
    scale = max(1, min(grey.shape) // 400)
    small = cv2.resize(
        grey, (grey.shape[1] // scale, grey.shape[0] // scale), interpolation=cv2.INTER_AREA
    )
    paper = cv2.medianBlur(small, 31)
    paper = cv2.resize(paper, (grey.shape[1], grey.shape[0]), interpolation=cv2.INTER_LINEAR).astype(
        np.float32
    )
    darker = np.clip((paper - grey) / np.maximum(paper, 1.0), 0.0, 1.0)
    scaled = (darker * 255).astype(np.uint8)
    threshold, _ = (
        cv2.threshold(scaled[scaled > 12], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if (scaled > 12).any()
        else (51, None)
    )
    return scaled > max(51, threshold)


class Frame:
    """The geometry of one line: along it and across it, whichever way it runs."""

    def __init__(self, vertical: bool, ink: np.ndarray):
        self.vertical = vertical
        self.ink = ink if vertical else ink.T

    def of(self, box: Box) -> tuple[tuple[int, int], tuple[int, int]]:
        if self.vertical:
            return (int(box.y), int(box.y2)), (int(box.x), int(box.x2))
        return (int(box.x), int(box.x2)), (int(box.y), int(box.y2))

    def box(self, along: tuple[int, int], across: tuple[int, int]) -> Box:
        if self.vertical:
            return Box(
                x=across[0],
                y=along[0],
                w=across[1] - across[0],
                h=along[1] - along[0],
            )
        return Box(
            x=along[0],
            y=across[0],
            w=along[1] - along[0],
            h=across[1] - across[0],
        )

    def patch(self, along: tuple[int, int], across: tuple[int, int]) -> np.ndarray:
        rows, columns = self.ink.shape
        return self.ink[
            max(0, along[0]) : max(0, min(rows, along[1])),
            max(0, across[0]) : max(0, min(columns, across[1])),
        ]

    def tight(self, along: tuple[int, int], across: tuple[int, int]):
        """The extent of the ink inside a rectangle, or None when it holds none."""
        patch = self.patch(along, across)
        if patch.size == 0:
            return None
        rows = np.flatnonzero(patch.any(axis=1))
        columns = np.flatnonzero(patch.any(axis=0))
        if rows.size == 0:
            return None
        top, left = max(0, along[0]), max(0, across[0])
        return (
            (top + int(rows[0]), top + int(rows[-1]) + 1),
            (left + int(columns[0]), left + int(columns[-1]) + 1),
        )


def valleys(profile: np.ndarray, parts: int) -> list[tuple[int, float]]:
    """Where to cut a profile into `parts`: the thinnest row near each even division."""
    height = len(profile)
    if height < 8 * parts:
        return []
    smooth = np.convolve(profile, np.ones(3) / 3, mode="same")
    mean = float(smooth.mean()) or 1.0
    cuts = []
    for division in range(1, parts):
        start = int(height * (division - 0.4) / parts)
        end = max(start + 1, int(height * (division + 0.4) / parts))
        row = start + int(np.argmin(smooth[start:end]))
        cuts.append((row, float(smooth[row]) / mean))
    return cuts


def columns_of(boxes: Sequence[Box], gap_ratio: float = GAP_RATIO, merge_ratio: float = MERGE_RATIO):
    """The box indices grouped into columns, right to left.

    A column is a run of boxes whose centres follow one another by no more than `gap_ratio` of the
    page's median character width; two runs merge when the white space between them is narrower than
    `merge_ratio` of that width, because a leaning line pauses without ending.
    """
    if not boxes:
        return []
    width = statistics.median(box.w for box in boxes)
    order = sorted(range(len(boxes)), key=lambda index: boxes[index].centre[0])
    runs: list[list[int]] = [[order[0]]]
    for index in order[1:]:
        if (
            boxes[index].centre[0] - boxes[runs[-1][-1]].centre[0]
            <= width * gap_ratio
        ):
            runs[-1].append(index)
        else:
            runs.append([index])
    merged: list[list[int]] = [runs[0]]
    for run in runs[1:]:
        here = statistics.mean(boxes[index].centre[0] for index in run)
        there = statistics.mean(boxes[index].centre[0] for index in merged[-1])
        if here - there < width * merge_ratio:
            merged[-1].extend(run)
        else:
            merged.append(run)
    merged.reverse()
    return merged


def pieces_of_ink(frame: Frame, along, across, height: float, width: float, column: int):
    """The runs of ink in a stretch of a line no detection covers, each as `(along, across)`."""
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
                cells.append({"along": tight[0], "across": tight[1], "column": column})
    return cells
