from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from PIL import Image

from glyph_cutter import detect
from glyph_cutter.schema import Box


@dataclass
class Item:
    name: str
    shape: list


class Stub:
    def __init__(self, outputs, names=("dets",), inputs=("pixel_values",)):
        self.outputs = outputs
        self.names = names
        self.inputs = inputs

    def get_inputs(self):
        return [Item(name, [1, 3, 1024, 1024]) for name in self.inputs]

    def get_outputs(self):
        return [Item(name, [1, 10, 5]) for name in self.names]

    def run(self, _names, _feed):
        return self.outputs

    def get_providers(self):
        return ["CPUExecutionProvider"]


def test_tile_origins_cover_the_axis():
    assert detect.tile_origins(1024) == [0]
    assert detect.tile_origins(1025) == [0, 896]
    assert detect.tile_origins(3000) == [0, 896, 1792, 2688]


def test_tiles_are_square_and_in_reading_order():
    found = detect.tiles(1100, 2200)
    assert len(found) == 2 * 3
    assert all(tile.w == tile.h == 1024 for tile in found)
    assert found[0].x == 0 and found[0].y == 0


def test_suppress_drops_a_duplicate():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    assert detect.suppress(boxes, scores, 0.5) == [0]


def test_decode_reads_dets_layout():
    stub = Stub([np.array([[[10, 20, 40, 60, 0.9], [100, 120, 140, 160, 0.5]]], dtype=np.float32)])
    detector = detect.Detector("unused", session=stub, format=None)
    boxes, scores = detector._decode(stub.outputs, 1024)
    assert boxes.shape == (2, 4)
    assert np.allclose(boxes[0], [10, 20, 40, 60])
    assert np.allclose(scores, [0.9, 0.5])


def test_decode_reads_rtdetr_layout():
    logits = np.zeros((1, 2, 1), dtype=np.float32)
    boxes = np.array([[[0.5, 0.5, 0.1, 0.1], [0.25, 0.25, 0.2, 0.2]]], dtype=np.float32)
    stub = Stub([logits, boxes], names=("logits", "pred_boxes"))
    detector = detect.Detector("unused", session=stub, format=None)
    boxes_out, _ = detector._decode(stub.outputs, 1000)
    assert np.allclose(boxes_out[0], [450, 450, 550, 550])


def test_boxes_maps_a_tile_detection_to_the_page():
    outputs = [np.array([[[100, 200, 300, 500, 0.9]]], dtype=np.float32)]
    detector = detect.Detector("unused", session=Stub(outputs), score=0.0, ink=0.0)
    page = Image.new("RGB", (1024, 1024), "white")
    found = detector.boxes(page)
    assert len(found) == 1
    assert found[0][0] == Box(100, 200, 200, 300)
    assert found[0][1] == pytest.approx(0.9)


def test_inked_drops_a_box_with_no_ink():
    page = Image.new("RGB", (100, 100), "white")
    assert detect.inked(page, [Box(10, 10, 20, 20)], share=0.4) == [True]
    page.paste((0, 0, 0), (42, 42, 48, 58))
    assert detect.inked(page, [Box(10, 10, 20, 20), Box(40, 40, 20, 20)], share=0.4) == [False, True]
