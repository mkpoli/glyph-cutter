# How it works

## Detect

A page is cut into 1024 px tiles with 128 px of overlap, padded with white at the right and bottom
edges. Each tile is one batch of the RT-DETR export. Detections are mapped back to page coordinates
and duplicates from overlapping tiles are removed by non-maximum suppression. A box must hold a
share of the page's typical ink, which drops boxes on show-through, stains and bare paper. A page
whose median box is under 50 px a side is detected again enlarged, because the detector knows
characters about 50 to 120 px a side best.

## Cut a page without a transcription

The page's detections are chained into columns, and a column into *cells*: a detection, the parts of
a detection tall enough for two or three characters, and unboxed ink between boxes. A candidate box
is a short run of neighbouring cells.

The framing model gives each candidate the probability that it frames one character. A column is cut
into the candidates whose odds multiply highest, with a cost for leaving a detection out. A candidate
that is not a detector box must be more likely than not to frame a character, so a doubtful cut
leaves the detector's box whole. A box the column leaves whole is then read alone by its shape, so
two small characters in a box, or two side by side, are cut apart.

This is the same segmenter Glyph Atlas uses for the Kyujanggak registers and other pages without a
transcription.

## Cut transcribed lines

The line's transcription gives the tokens and their order; the page gives the ink. A token may take
one cell, a run of up to three cells, or no ink at all. The alignment is the cheapest path through
tokens and cells. A placement costs the classifier's negative log probability of the token over the
candidate crop, plus the framing model's penalty for a box that does not look like one character.

A second model, the gate, accepts a placement for output from what the alignment knows about it: how
the classifier read the crop, how much dearer the alternatives were, and how its neighbours were
placed. The gate was fitted for 98.5% precision; `--all` keeps the placements it rejects.

## Square crops

A character box is rarely square. The crop expands the shorter side about the box's centre, adds
padding, shifts inside the page where it fits, and fills with the page's paper colour where it does
not. The result is resized to `--size` with a Lanczos filter. The manifest keeps both the character
box and the square actually cut.

## What it does not do

It does not read the characters. A crop is one character only when the detector and the cutter are
right; the manifest does not carry a transcription. It does not remove ruby, punctuation or
decoration that the detector may have boxed, and it does not segment a page into text blocks.
