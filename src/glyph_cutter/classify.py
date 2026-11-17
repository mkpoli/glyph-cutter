"""Source-class probabilities from an ONNX character classifier.

One crop is one written character. The crop is converted to grey, padded to a square with white so
its shape is kept, resized to the side the export takes, replicated to three channels, scaled to
[0, 1] and normalised with the ImageNet statistics averaged over the channels. The model returns a
probability for every class of `classes.json`, or logits over them.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SIZE = 96
PAD = 255
MEAN = 0.449
STD = 0.226
OTHER = "other"
CLASSES_NAME = "classes.json"
CHUNK = 32


class ClassifierError(RuntimeError):
    """An export or a class list that cannot be used."""


def grey(crop: Image.Image) -> Image.Image:
    """The crop as one channel on white."""
    image = crop
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        ground = Image.new("RGBA", image.size, (PAD, PAD, PAD, 255))
        ground.alpha_composite(image.convert("RGBA"))
        image = ground
    return image.convert("L")


def square(crop: Image.Image, *, pad: int = PAD) -> Image.Image:
    """The crop padded with `pad` to a square, centred, its aspect ratio unchanged."""
    side = max(crop.size)
    canvas = Image.new(crop.mode, (side, side), pad)
    canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return canvas


def preprocess(crop: Image.Image, *, size: int = SIZE) -> Image.Image:
    """A crop as the grey `size` x `size` square the model takes."""
    if crop.width < 1 or crop.height < 1:
        raise ValueError(f"crop must have a positive size, got {crop.width}x{crop.height}")
    return square(grey(crop)).resize((size, size), Image.Resampling.BILINEAR)


def crop_array(crop: Image.Image, *, size: int = SIZE) -> np.ndarray:
    """A crop as the (1, 3, `size`, `size`) float32 tensor the model takes."""
    array = np.asarray(preprocess(crop, size=size), dtype=np.float32) / np.float32(255.0)
    array = (array - np.float32(MEAN)) / np.float32(STD)
    return np.ascontiguousarray(np.repeat(array[None], 3, axis=0)[None])


def read_classes(path: Path | str) -> list[str]:
    """The class order of a `classes.json`, which is a list or an object with a `classes` key."""
    file = Path(path)
    document = json.loads(file.read_text(encoding="utf-8"))
    classes = document.get("classes") if isinstance(document, dict) else document
    if not isinstance(classes, list) or not classes or not all(isinstance(name, str) for name in classes):
        raise ClassifierError(
            f"{file}: expected a list of class names, or an object with a 'classes' list"
        )
    if len(set(classes)) != len(classes):
        raise ClassifierError(f"{file}: the class list repeats a name")
    return list(classes)


def classes_for(onnx_path: Path | str, classes: Sequence[str] | Path | str | None = None) -> list[str]:
    """The class list of an export: what the caller gave, or `classes.json` near the export."""
    if classes is not None and not isinstance(classes, (str, Path)):
        return [str(name) for name in classes]
    if isinstance(classes, (str, Path)):
        return read_classes(classes)
    export = Path(onnx_path)
    for candidate in (export.parent / CLASSES_NAME, export.parent.parent / CLASSES_NAME):
        if candidate.is_file():
            return read_classes(candidate)
    raise ClassifierError(
        f"no {CLASSES_NAME} beside {export} or its parent; pass the class list to Classifier"
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - values.max(axis=-1, keepdims=True)
    exponent = np.exp(values)
    return exponent / exponent.sum(axis=-1, keepdims=True)


class Classifier:
    """A classifier over an exported ONNX model and its class list."""

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        classes: Sequence[str] | Path | str | None = None,
        session: Any | None = None,
        providers: Sequence[str] | None = None,
        size: int | None = None,
    ) -> None:
        if size is not None and size < 1:
            raise ValueError(f"size must be positive, got {size}")
        self.onnx_path = Path(onnx_path)
        self.classes = classes_for(self.onnx_path, classes)
        self._session = session if session is not None else self._build_session(providers)
        self.size = int(size or self._input_size())

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

    def _input_size(self) -> int:
        inputs = getattr(self._session, "get_inputs", None)
        shape = inputs()[0].shape if inputs is not None and inputs() else []
        side = shape[2] if len(shape) == 4 else None
        return side if isinstance(side, int) and side > 0 else SIZE

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

    def probabilities(self, crop: Image.Image | np.ndarray) -> np.ndarray:
        return self.probabilities_many([crop])[0]

    def probabilities_many(self, crops: Iterable[Image.Image | np.ndarray]) -> np.ndarray:
        """The class probabilities of crops, as an (n, classes) array."""
        images = [self._pixels(crop) for crop in crops]
        if not images:
            return np.zeros((0, len(self.classes)), dtype=np.float64)
        width = self.batch_width()
        if width == 1 or len(images) <= width:
            return self._probabilities_of(np.concatenate(images, axis=0))
        parts = [
            self._probabilities_of(np.concatenate(images[start : start + width], axis=0))
            for start in range(0, len(images), width)
        ]
        return np.concatenate(parts, axis=0)

    def batch_width(self) -> int:
        """How many crops one run of the exported graph is given: its fixed batch, or `CHUNK`."""
        described = getattr(self._session, "get_inputs", None)
        if described is None:
            return 1
        for item in described():
            if item.name != self.input_name:
                continue
            shape = list(item.shape or [])
            if not shape:
                break
            first = shape[0]
            if isinstance(first, int) and first > 0:
                return first
            return CHUNK
        return 1

    def _probabilities_of(self, batch: np.ndarray) -> np.ndarray:
        outputs = self._session.run(None, {self.input_name: batch})
        described = getattr(self._session, "get_outputs", None)
        names = [item.name for item in described()] if described is not None else []
        if len(names) != len(outputs):
            names = [f"output{index}" for index in range(len(outputs))]
        lowered = {name.lower(): np.asarray(value) for name, value in zip(names, outputs, strict=True)}
        if "probs" in lowered:
            values = np.asarray(lowered["probs"], dtype=np.float64)
        else:
            logits = next((value for name, value in lowered.items() if "logit" in name), None)
            if logits is None:
                logits = np.asarray(outputs[0], dtype=np.float64)
            values = softmax(logits)
        values = values.reshape(batch.shape[0], -1)
        if values.shape[1] != len(self.classes):
            raise ClassifierError(
                f"{self.onnx_path}: the model returns {values.shape[1]} classes and "
                f"{CLASSES_NAME} names {len(self.classes)}"
            )
        return values

    def scores(self, crop: Image.Image | np.ndarray) -> dict[str, float]:
        values = self.probabilities(crop)
        return {name: float(value) for name, value in zip(self.classes, values, strict=True)}

    def _pixels(self, crop: Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(crop, np.ndarray):
            array = np.asarray(crop, dtype=np.float32)
            if array.ndim == 2:
                return crop_array(Image.fromarray(array.astype(np.uint8), mode="L"), size=self.size)
            raise ValueError(f"a numpy crop must be a 2-D grey array, got shape {array.shape}")
        return crop_array(crop, size=self.size)
