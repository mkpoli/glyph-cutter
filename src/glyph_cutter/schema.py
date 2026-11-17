"""The small data model the cutter passes between its stages."""

from __future__ import annotations

from dataclasses import dataclass, field

KINDS = ("box", "part", "merge", "ink", "mixed")


@dataclass(frozen=True, slots=True)
class Box:
    """A rectangle in page pixels."""

    x: float
    y: float
    w: float
    h: float

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def centre(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def intersection(self, other: Box) -> float:
        return max(0.0, min(self.x2, other.x2) - max(self.x, other.x)) * max(
            0.0, min(self.y2, other.y2) - max(self.y, other.y)
        )

    def iou(self, other: Box) -> float:
        shared = self.intersection(other)
        union = self.area + other.area - shared
        return shared / union if union > 0 else 0.0

    def contains(self, point: tuple[float, float]) -> bool:
        return self.x <= point[0] <= self.x2 and self.y <= point[1] <= self.y2

    def clipped(self, width: int, height: int) -> Box | None:
        left, top = max(0.0, self.x), max(0.0, self.y)
        right, bottom = min(float(width), self.x2), min(float(height), self.y2)
        if right <= left or bottom <= top:
            return None
        return Box(left, top, right - left, bottom - top)

    def rounded(self) -> tuple[int, int, int, int]:
        """`(x, y, w, h)` as non-negative integers, at least one pixel a side."""
        x, y = round(self.x), round(self.y)
        return x, y, max(1, round(self.w)), max(1, round(self.h))

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> Box:
        return cls(x1, y1, x2 - x1, y2 - y1)


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected character box and the detector's score for it."""

    box: Box
    score: float

    @property
    def centre(self) -> tuple[float, float]:
        return self.box.centre


@dataclass(frozen=True, slots=True)
class Token:
    """One unit of a transcription the lattice may place."""

    text: str
    start: int = 0
    end: int = 0
    code_points: frozenset[str] = frozenset()
    kind: str = "char"  # "char", "gap", "mark" or "punctuation"
    script: str = "unknown"


@dataclass(frozen=True, slots=True)
class Line:
    """A line as a transcription gives it: a box, a reading direction and its text."""

    id: str
    text: str
    vertical: bool = True
    box: Box | None = None
    page_id: str = ""
    seq: int = 0


@dataclass(slots=True)
class Placement:
    """Where one token was placed, and what the models said about it."""

    token: Token
    box: Box | None = None
    kind: str = "skip"
    cost: float = 0.0
    features: dict[str, float] = field(default_factory=dict)
    probability: float = 0.0
    accepted: bool = False


@dataclass(frozen=True, slots=True)
class Piece:
    """One character box of a page that has no transcription."""

    box: Box
    kind: str
    probability: float
    detections: tuple[int, ...] = ()
