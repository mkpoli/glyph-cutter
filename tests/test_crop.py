from __future__ import annotations

from pathlib import Path

from PIL import Image

from glyph_cutter.crop import Manifest, crop_square, paper_colour, square_box
from glyph_cutter.schema import Box


def test_square_box_adds_padding_on_both_axes():
    square = square_box(Box(90, 90, 20, 40), 200, 200, padding=0.1)
    assert square.w == square.h == 48
    assert (square.x, square.y) == (76, 86)


def test_square_box_stays_inside_the_page():
    square = square_box(Box(0, 0, 20, 20), 100, 100, padding=0.5)
    assert (square.x, square.y, square.w) == (0, 0, 40)


def test_square_box_centres_when_larger_than_the_page():
    square = square_box(Box(0, 0, 20, 20), 30, 30, padding=1.0)
    assert square.w == 60
    assert (square.x, square.y) == (-15, -15)


def test_crop_square_fills_outside_the_page_with_paper():
    page = Image.new("RGB", (30, 30), (255, 255, 255))
    page.paste((0, 0, 0), (0, 0, 10, 10))
    image, square = crop_square(page, Box(0, 0, 10, 10), padding=1.0, size=0)
    assert image.size == (30, 30)
    assert square.w == 30
    assert image.getpixel((0, 0)) == (0, 0, 0)
    assert image.getpixel((29, 29)) == (255, 255, 255)


def test_crop_square_resizes_to_the_requested_side():
    page = Image.new("RGB", (100, 100), "white")
    image, _ = crop_square(page, Box(40, 40, 10, 20), padding=0.0, size=64)
    assert image.size == (64, 64)


def test_paper_colour_is_the_median():
    page = Image.new("RGB", (4, 4), (250, 240, 230))
    page.paste((10, 10, 10), (0, 0, 1, 1))
    assert paper_colour(page) == (250, 240, 230)


def test_manifest_writes_jsonl_and_run(tmp_path: Path):
    manifest = Manifest(tmp_path)
    manifest.add({"index": 1, "text": "字"})
    manifest.close({"crops": 1})
    assert (tmp_path / "manifest.jsonl").read_text(encoding="utf-8") == '{"index": 1, "text": "字"}\n'
    assert '"crops": 1' in (tmp_path / "run.json").read_text(encoding="utf-8")
