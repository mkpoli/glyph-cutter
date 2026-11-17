"""Character boxes on a page image, from an RT-DETR ONNX export.

A page is cut into 1024x1024 tiles with 128 px of overlap, the geometry the detector's training data
build uses, and the tiles at the right and bottom edge are padded with white. A detection is mapped
from tile coordinates to page coordinates, and duplicates that two overlapping tiles both report are
removed by score-ordered non-maximum suppression over the whole page. A kept box then has to hold
ink, which drops boxes on show-through, stains and bare paper.

The model takes one float32 tensor, `pixel_values`, of shape (1, 3, 1024, 1024): RGB, scaled to
[0, 1], normalised with the ImageNet mean and standard deviation, and no resizing. It returns either
the RT-DETR pair `logits` and `pred_boxes` (centre form, normalised to the tile), or one `dets`
tensor of pixel `x1, y1, x2, y2`, score and an optional label.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .schema import Box

TILE = 1024
OVERLAP = 128
STRIDE = TILE - OVERLAP
PAD = 255

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

MIN_SIZE = 2
CONTAIN = 0.9
INK = 0.4
INK_PERCENTILE = 0.5
PAPER_PERCENTILE = 90
INK_REFERENCE = 75
PART = 0.7
MEETS = 0.3
KNOWN = 81.0
SMALL = 50.0
MOST = 2.0
LARGE = 118.0
CLEAR = 0.1
BLOCK = 3


def tile_origins(length: int, tile: int = TILE, overlap: int = OVERLAP) -> list[int]:
    """Origins along one axis of the tiles that cover `length` pixels."""
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    stride = tile - overlap
    if stride <= 0:
        raise ValueError(f"overlap {overlap} must be smaller than the tile size {tile}")
    if length <= tile:
        return [0]
    return [index * stride for index in range(math.ceil((length - overlap) / stride))]


def tiles(width: int, height: int, tile: int = TILE, overlap: int = OVERLAP) -> list[Box]:
    """Every tile of a `width` x `height` page, in reading order."""
    if width <= 0 or height <= 0:
        raise ValueError(f"page must have a positive size, got {width}x{height}")
    return [
        Box(float(x), float(y), float(tile), float(tile))
        for y in tile_origins(height, tile, overlap)
        for x in tile_origins(width, tile, overlap)
    ]


def meets(a: Box, b: Box) -> bool:
    """Whether two boxes share at least one pixel."""
    return a.x < b.x2 and b.x < a.x2 and a.y < b.y2 and b.y < a.y2


def open_page(image: Image.Image | str | Path) -> Image.Image:
    """The page as an RGB image."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as handle:
            return handle.convert("RGB")
    return image if image.mode == "RGB" else image.convert("RGB")


def tile_array(page: Image.Image, tile: Box) -> np.ndarray:
    """One tile as a (1, 3, tile, tile) float32 batch, padded with white outside the page."""
    x, y, w, h = int(tile.x), int(tile.y), int(tile.w), int(tile.h)
    canvas = Image.new("RGB", (w, h), (PAD, PAD, PAD))
    right, bottom = min(x + w, page.width), min(y + h, page.height)
    left, top = max(x, 0), max(y, 0)
    if right > left and bottom > top:
        canvas.paste(page.crop((left, top, right, bottom)), (left - x, top - y))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    array = (array - np.asarray(MEAN, dtype=np.float32)) / np.asarray(STD, dtype=np.float32)
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None])


def box_ink(grey: np.ndarray, box: Box) -> float:
    """How far the darkest pixels of `box` fall below its paper, in grey levels."""
    x, y, w, h = int(round(box.x)), int(round(box.y)), int(round(box.w)), int(round(box.h))
    region = grey[max(0, y) : max(0, y + h), max(0, x) : max(0, x + w)]
    if region.size == 0:
        return 0.0
    paper, darkest = np.percentile(region, (PAPER_PERCENTILE, INK_PERCENTILE))
    return max(0.0, float(paper) - float(darkest))


def inked(page: Image.Image, boxes: Sequence[Box], share: float = INK) -> list[bool]:
    """Whether each box holds at least `share` of the page's typical ink."""
    if not boxes or share <= 0:
        return [True] * len(boxes)
    from PIL import ImageFilter

    grey = np.asarray(page.convert("L").filter(ImageFilter.MedianFilter(3)), dtype=np.float32)
    values = [box_ink(grey, box) for box in boxes]
    typical = float(np.percentile(values, INK_REFERENCE))
    return [value >= share * typical for value in values]


def _area(boxes: np.ndarray) -> np.ndarray:
    return np.clip(boxes[..., 2] - boxes[..., 0], 0.0, None) * np.clip(
        boxes[..., 3] - boxes[..., 1], 0.0, None
    )


def _intersection(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    left = np.maximum(box[0], others[:, 0])
    top = np.maximum(box[1], others[:, 1])
    right = np.minimum(box[2], others[:, 2])
    bottom = np.minimum(box[3], others[:, 3])
    return np.clip(right - left, 0.0, None) * np.clip(bottom - top, 0.0, None)


def iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    inter = _intersection(box, others)
    union = _area(box) + _area(others) - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


def suppress(
    boxes: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    clipped: np.ndarray | None = None,
    contain: float = CONTAIN,
) -> list[int]:
    """Indexes kept by greedy non-maximum suppression, highest score first."""
    order = np.argsort(-scores, kind="stable")
    if clipped is not None:
        order = order[np.argsort(clipped[order], kind="stable")]
    keep: list[int] = []
    while order.size:
        best = int(order[0])
        keep.append(best)
        if order.size == 1:
            break
        rest = order[1:]
        drop = iou(boxes[best], boxes[rest]) > threshold
        if clipped is not None:
            smaller = np.minimum(_area(boxes[best]), _area(boxes[rest]))
            inside = _intersection(boxes[best], boxes[rest]) > contain * np.maximum(smaller, 1e-6)
            drop |= clipped[rest] & inside
        order = rest[~drop]
    return keep


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def cxcywh_to_xyxy(boxes: np.ndarray, size: int) -> np.ndarray:
    centre_x, centre_y, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack(
        [
            (centre_x - width / 2) * size,
            (centre_y - height / 2) * size,
            (centre_x + width / 2) * size,
            (centre_y + height / 2) * size,
        ],
        axis=1,
    ).astype(np.float32)


def clipped_by_border(boxes: np.ndarray, tile: Box, width: int, height: int) -> np.ndarray:
    """Which detections touch a tile border that lies inside the page."""
    flags = np.zeros(len(boxes), dtype=bool)
    if boxes.size == 0:
        return flags
    edge = np.float32(0.5)
    if tile.x > 0:
        flags |= boxes[:, 0] <= edge
    if tile.y > 0:
        flags |= boxes[:, 1] <= edge
    if tile.x + tile.w < width:
        flags |= boxes[:, 2] >= np.float32(tile.w) - edge
    if tile.y + tile.h < height:
        flags |= boxes[:, 3] >= np.float32(tile.h) - edge
    return flags


def post_process(
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    score: float = 0.3,
    nms: float | None = None,
    max_per_tile: int = 1500,
    clipped: np.ndarray | None = None,
    contain: float = CONTAIN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Keep the detections that pass `score`, cap them and optionally suppress overlaps."""
    keep = np.flatnonzero(scores >= score)
    if keep.size > max_per_tile:
        keep = keep[np.argsort(-scores[keep], kind="stable")[:max_per_tile]]
    boxes, scores = boxes[keep], scores[keep]
    cut = None if clipped is None else clipped[keep]
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), cut
    if nms is not None:
        order = np.asarray(suppress(boxes, scores, nms, clipped=cut, contain=contain), dtype=int)
        boxes, scores = boxes[order], scores[order]
        cut = None if cut is None else cut[order]
    return boxes, scores, cut


def _meet(box: Box, other: Box) -> float:
    return box.intersection(other)


def enlargement(boxes: Sequence[Box]) -> float:
    """How many times a page with these boxes is enlarged for the detector; one where it is not."""
    if not boxes:
        return 1.0
    sides = np.sqrt([box.w * box.h for box in boxes])
    median = float(np.median(sides))
    if median >= SMALL:
        return 1.0
    return max(1.0, min(MOST, KNOWN / median, LARGE / float(np.percentile(sides, 90))))


def _share(box: Box, other: Box) -> float:
    return _meet(box, other) / (box.w * box.h)


def _apart(boxes: Sequence[Box]) -> int:
    kept: list[Box] = []
    for box in sorted(boxes, key=lambda box: -box.w * box.h):
        if all(max(_share(box, other), _share(other, box)) < MEETS for other in kept):
            kept.append(box)
    return len(kept)


def _to_box(box: np.ndarray, width: int, height: int) -> Box | None:
    left, top = max(0.0, float(box[0])), max(0.0, float(box[1]))
    right, bottom = min(float(box[2]), float(width)), min(float(box[3]), float(height))
    if right - left < MIN_SIZE or bottom - top < MIN_SIZE:
        return None
    x, y = round(left), round(top)
    return Box(float(x), float(y), float(max(1, round(right) - x)), float(max(1, round(bottom) - y)))


class Detector:
    """A detector over an exported ONNX model.

    `score` is the lowest score a detection keeps, `nms` the IoU above which a lower-scoring
    duplicate is dropped, `max_per_tile` the number of detections one tile may contribute, and
    `ink` the share of the page's typical ink a kept box must hold (0 keeps every box).
    """

    def __init__(
        self,
        onnx_path: str | Path,
        score: float = 0.3,
        nms: float = 0.5,
        max_per_tile: int = 1500,
        ink: float = INK,
        *,
        session: Any | None = None,
        providers: Sequence[str] | None = None,
        tile: int = TILE,
        overlap: int = OVERLAP,
        format: str | None = None,
    ) -> None:
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"score must lie in [0, 1], got {score}")
        if not 0.0 <= nms <= 1.0:
            raise ValueError(f"nms must lie in [0, 1], got {nms}")
        if max_per_tile < 1:
            raise ValueError(f"max_per_tile must be positive, got {max_per_tile}")
        if not 0.0 <= ink <= 1.0:
            raise ValueError(f"ink must lie in [0, 1], got {ink}")
        if format not in (None, "rtdetr", "dets"):
            raise ValueError(f"unknown output format {format!r}")
        self.onnx_path = Path(onnx_path)
        self.score = float(score)
        self.nms = float(nms)
        self.max_per_tile = int(max_per_tile)
        self.ink = float(ink)
        self.tile = int(tile)
        self.overlap = int(overlap)
        self.format = format
        self._session = session if session is not None else self._build_session(providers)

    def _build_session(self, providers: Sequence[str] | None) -> Any:
        import onnxruntime as ort

        wanted = list(providers or ("CUDAExecutionProvider", "CPUExecutionProvider"))
        offered = ort.get_available_providers()
        usable = [name for name in wanted if name in offered]
        if not usable:
            raise RuntimeError(f"onnxruntime offers {offered}, none of {wanted}")
        preload = getattr(ort, "preload_dlls", None)
        if preload is not None and any(name.startswith("CUDA") for name in usable):
            with contextlib.suppress(ImportError, OSError, RuntimeError):
                preload()
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(str(self.onnx_path), sess_options=options, providers=usable)

    @property
    def input_name(self) -> str:
        inputs = getattr(self._session, "get_inputs", None)
        if inputs is None:
            return "pixel_values"
        described = inputs()
        return described[0].name if described else "pixel_values"

    def providers(self) -> list[str]:
        described = getattr(self._session, "get_providers", None)
        return list(described()) if described is not None else []

    def boxes(self, image: Image.Image | str | Path) -> list[tuple[Box, float]]:
        """Every character box on the page, in page coordinates, highest score first."""
        page = open_page(image)
        return self._collect(page, tiles(page.width, page.height, self.tile, self.overlap))

    def boxes_sized(self, image: Image.Image | str | Path) -> list[tuple[Box, float]]:
        """The page's boxes, found enlarged when its writing is smaller than the detector knows."""
        page = open_page(image)
        own = self.boxes(page)
        scale = enlargement([box for box, _ in own])
        return own if scale == 1.0 else self.boxes_at(page, scale, own=own)

    def boxes_at(
        self,
        image: Image.Image | str | Path,
        scale: float,
        fill: Sequence[float] = (),
        *,
        own: list[tuple[Box, float]] | None = None,
    ) -> list[tuple[Box, float]]:
        """The boxes found on the page resized by `scale`, in the page's own coordinates."""
        if scale <= 0 or any(other <= 0 for other in fill):
            raise ValueError(f"a scale must be positive, got {scale} and {tuple(fill)}")
        page = open_page(image)
        own = self.boxes(page) if own is None else own
        kept = list(own) if scale == 1.0 else self._joined(own, self._resized(page, scale))
        for other in fill:
            standing = [box for box, _ in kept]
            kept += [
                (box, score)
                for box, score in (own if other == 1.0 else self._resized(page, other))
                if sum(_meet(box, held) for held in standing) < CLEAR * box.w * box.h
                and not any(_share(held, box) >= PART for held in standing)
            ]
        kept.sort(key=lambda pair: (-pair[1], pair[0].y, pair[0].x, pair[0].h, pair[0].w))
        return kept

    def _resized(self, page: Image.Image, scale: float) -> list[tuple[Box, float]]:
        resized = page.resize(
            (round(page.width * scale), round(page.height * scale)), Image.Resampling.LANCZOS
        )
        found = []
        for box, score in self.boxes(resized):
            mapped = _to_box(
                np.asarray([box.x, box.y, box.x2, box.y2]) / scale, page.width, page.height
            )
            if mapped is not None:
                found.append((mapped, score))
        return found

    @staticmethod
    def _joined(own: list[tuple[Box, float]], found: list[tuple[Box, float]]) -> list[tuple[Box, float]]:
        found = [
            (box, score)
            for box, score in found
            if _apart([other for other, _ in own if _share(other, box) >= PART]) < BLOCK
        ]
        whole = [
            (box, score)
            for box, score in found
            if not any(
                other.w * other.h > box.w * box.h and _share(box, other) >= PART for other, _ in found
            )
        ]
        stands = [
            (box, score)
            for box, score in own
            if sum(_meet(box, other) for other, _ in whole) < MEETS * box.w * box.h
        ]
        whole = [
            (box, score)
            for box, score in whole
            if not any(
                other.w * other.h > box.w * box.h and _share(box, other) >= PART for other, _ in stands
            )
        ]
        return whole + stands

    def _collect(self, page: Image.Image, wanted: Iterable[Box]) -> list[tuple[Box, float]]:
        """Run every tile, map the detections to page coordinates and suppress duplicates."""
        found: list[np.ndarray] = []
        scores: list[float] = []
        clipped: list[bool] = []
        for tile in wanted:
            boxes, tile_scores = self._detect_tile(page, tile)
            cut = clipped_by_border(boxes, tile, page.width, page.height)
            for box, score, is_cut in zip(boxes, tile_scores, cut, strict=True):
                found.append([box[0] + tile.x, box[1] + tile.y, box[2] + tile.x, box[3] + tile.y])
                scores.append(float(score))
                clipped.append(bool(is_cut))
        if not found:
            return []
        stack = np.asarray(found, dtype=np.float32)
        score_array = np.asarray(scores, dtype=np.float32)
        stack[:, 0] = np.clip(stack[:, 0], 0.0, page.width)
        stack[:, 1] = np.clip(stack[:, 1], 0.0, page.height)
        stack[:, 2] = np.clip(stack[:, 2], 0.0, page.width)
        stack[:, 3] = np.clip(stack[:, 3], 0.0, page.height)
        keep = suppress(stack, score_array, self.nms, clipped=np.asarray(clipped, dtype=bool))
        out: list[tuple[Box, float]] = []
        for index in keep:
            box = _to_box(stack[index], page.width, page.height)
            if box is not None:
                out.append((box, float(score_array[index])))
        kept = inked(page, [box for box, _ in out], self.ink)
        out = [pair for pair, keep_me in zip(out, kept, strict=True) if keep_me]
        out.sort(key=lambda pair: (-pair[1], pair[0].y, pair[0].x, pair[0].h, pair[0].w))
        return out

    def _detect_tile(self, page: Image.Image, tile: Box) -> tuple[np.ndarray, np.ndarray]:
        outputs = self._session.run(None, {self.input_name: tile_array(page, tile)})
        boxes, scores = self._decode(outputs, int(max(tile.w, tile.h)))
        if boxes.size == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        boxes = boxes.astype(np.float32)
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0.0, tile.w)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0.0, tile.h)
        boxes, scores, _ = post_process(
            boxes, scores, score=self.score, max_per_tile=self.max_per_tile
        )
        return boxes.astype(np.float32), scores.astype(np.float32)

    def _output_names(self, outputs: Sequence[np.ndarray]) -> list[str]:
        described = getattr(self._session, "get_outputs", None)
        if described is not None:
            names = [item.name for item in described()]
            if len(names) == len(outputs):
                return names
        return [f"output{index}" for index in range(len(outputs))]

    def _decode(self, outputs: Sequence[np.ndarray], size: int) -> tuple[np.ndarray, np.ndarray]:
        """Boxes on the tile and their scores, in the layout `format` names or the model implies."""
        names = self._output_names(outputs)
        tensors = {name: np.asarray(value) for name, value in zip(names, outputs, strict=True)}
        lowered = {name.lower(): name for name in names}
        logits_name = next((n for k, n in lowered.items() if "logit" in k), None)
        boxes_name = next((n for k, n in lowered.items() if "box" in k), None)
        dets_name = next((n for k, n in lowered.items() if "det" in k), None)
        layout = self.format
        if layout is None:
            if logits_name is not None and boxes_name is not None:
                layout = "rtdetr"
            elif dets_name is not None:
                layout = "dets"
            elif len(outputs) == 2:
                layout = "rtdetr"
            elif len(outputs) == 1 and np.asarray(outputs[0]).shape[-1] in (5, 6):
                layout = "dets"
            else:
                raise ValueError(f"cannot tell the output layout of {names}")
        if layout == "rtdetr":
            logits = tensors[logits_name] if logits_name else np.asarray(outputs[0])
            raw = tensors[boxes_name] if boxes_name else np.asarray(outputs[-1])
            scores = _sigmoid(np.asarray(logits, dtype=np.float32)[0])
            scores = scores.max(axis=-1) if scores.ndim > 1 else scores
            boxes = cxcywh_to_xyxy(np.asarray(raw, dtype=np.float32)[0], size)
        else:
            dets = np.asarray(tensors[dets_name] if dets_name else outputs[0], dtype=np.float32)[0]
            boxes, scores = dets[:, :4], dets[:, 4]
        return boxes, np.asarray(scores, dtype=np.float32)
