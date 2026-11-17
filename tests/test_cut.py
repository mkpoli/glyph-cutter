from __future__ import annotations

from glyph_cutter import cut, segment
from glyph_cutter.ink import columns_of
from glyph_cutter.schema import Box, Detection, Line, Piece


def test_tokens_keep_kana_and_mark_whitespace():
    tokens = cut.tokens_of("あ い")
    assert [token.text for token in tokens] == ["あ", "", "い"]
    assert [token.kind for token in tokens] == ["char", "gap", "char"]
    assert tokens[0].code_points == {"あ"}


def test_tokens_attach_a_combining_mark():
    tokens = cut.tokens_of("か\u3099")
    assert len(tokens) == 1
    assert tokens[0].text == "が"
    assert tokens[0].code_points == {"が"}


def test_columns_group_by_x_and_read_right_to_left():
    boxes = [Box(10, 10, 10, 10), Box(50, 10, 10, 10)]
    groups = columns_of(boxes)
    assert groups == [[1], [0]]


def test_assign_page_keeps_a_detection_in_the_smallest_line():
    lines = [
        Line(id="a", text="一", box=Box(0, 0, 100, 100)),
        Line(id="b", text="二", box=Box(0, 0, 20, 20)),
    ]
    detections = [Detection(Box(5, 5, 5, 5), 0.9)]
    own = cut.assign_page(lines, detections)
    assert own["b"] == detections
    assert own["a"] == []


def test_assign_page_sorts_a_vertical_line_top_to_bottom():
    line = Line(id="a", text="一二", vertical=True, box=Box(0, 0, 50, 100))
    detections = [Detection(Box(5, 50, 10, 10), 0.9), Detection(Box(5, 5, 10, 10), 0.9)]
    own = cut.assign_page([line], detections)
    assert [detection.box.y for detection in own["a"]] == [5, 50]


def test_reading_order_rtl_starts_at_the_right_column():
    pieces = [
        Piece(Box(10, 0, 20, 20), "box", 0.9),
        Piece(Box(100, 20, 20, 20), "box", 0.9),
        Piece(Box(100, 0, 20, 20), "box", 0.9),
    ]
    assert [piece.box.x for piece in segment.ordered(pieces, "rtl")] == [100, 100, 10]
    assert [piece.box.x for piece in segment.ordered(pieces, "ltr")] == [10, 100, 100]


def test_reading_order_ltr_reads_top_to_bottom():
    pieces = [
        Piece(Box(100, 20, 20, 20), "box", 0.9),
        Piece(Box(10, 0, 20, 20), "box", 0.9),
    ]
    assert [piece.box.y for piece in segment.ordered(pieces, "ltr")] == [0, 20]
