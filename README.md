# glyph-cutter

Cut square character crops out of scanned classical books. Give it a PDF, page images or a directory
of page images; it finds the characters, cuts each one to a square and writes a manifest beside the
crops. No transcription is needed.

The detector, classifier and cutter models come from the [Glyph Atlas](https://github.com/mkpoli/glyph-atlas)
project, where they were trained and measured. This project packages them as a standalone tool.

## Install

```sh
git clone https://github.com/mkpoli/glyph-cutter
cd glyph-cutter
uv sync --extra cpu        # or --extra gpu for CUDA
uv run glyph-cutter fetch  # about 200 MB of model weights
```

With pip:

```sh
pip install "glyph-cutter[cpu] @ git+https://github.com/mkpoli/glyph-cutter"
glyph-cutter fetch
```

Python 3.12 or later. `--extra gpu` installs `onnxruntime-gpu`; install one of the two extras, not
both.

## Cut a book

```sh
uv run glyph-cutter cut book.pdf --out crops/book
uv run glyph-cutter cut scans/ --out crops/book --dpi 300
uv run glyph-cutter cut page-01.jpg page-02.jpg --out crops/pages
```

Each crop is square, centred on the detected character, with a little paper around it, and resized
to `--size` (384 px by default). The page is searched for characters with no transcription. A page
of small writing is detected again enlarged when needed.

Useful options:

| Option | Meaning |
| --- | --- |
| `--reading rtl` | number crops right to left, then top to bottom (default; classical CJK books) |
| `--reading ltr` | number crops left to right, then top to bottom |
| `--size 0` | keep the square at source resolution instead of resizing |
| `--padding 0.15` | leave more paper around each character |
| `--format jpeg` | write JPEG instead of PNG |
| `--pages 1-20` | read a page range |
| `--max-pages 10` | stop after ten pages |
| `--device cpu` | run on the CPU |
| `--score`, `--nms` | detector score cutoff and duplicate suppression |

## Cut transcribed lines

If a book has line boxes and text, pass them as JSON or JSONL and the cutter places every
transcribed character on the page's ink:

```sh
uv run glyph-cutter cut book.pdf --mode lines --lines transcript.jsonl --out crops/book
```

A record gives a line box in page pixels, its reading direction and its text:

```json
{"source": "book.pdf", "page": 3, "box": [412, 180, 620, 1210], "vertical": true, "text": "東海道五十三次"}
```

`file` may stand in for `source` for an image input. `page` is 1-based. A placement the acceptance
model rejects is left out unless `--all` is given. This mode suits a transcription aligned to the
same page images; it needs the whole line box, not only its text.

## Output

```text
crops/book/
  manifest.jsonl
  run.json
  crops/
    book/
      p0001/
        0001.png
        0002.png
```

`manifest.jsonl` has one row per crop:

```json
{"mode":"page","source":"book.pdf","source_id":"book","page":1,"page_size":[2480,3508],
 "file":"crops/book/p0001/0001.png","crop":[480,212,420],"size":[384,384],
 "index":1,"box":[512,240,356,364],"probability":0.94,"kind":"box","detections":[17]}
```

`box` is the detected character, `[x, y, w, h]`; `crop` is the square actually cut, `[x, y, side]`.
`run.json` records the options, model checksums and crop count of the run.

## Models

`glyph-cutter fetch` writes five files to `~/.cache/glyph-cutter/models` (or `$GLYPH_CUTTER_MODELS`):

| File | Role |
| --- | --- |
| `detector.onnx` | RT-DETR character detector |
| `classifier.onnx`, `classes.json` | character classifier used to score candidate boxes |
| `frame.json` | does a candidate box frame one character |
| `gate.json` | accept a transcribed placement |

The files are checked against their SHA-256 on every run. A release with the weights is at
[releases](https://github.com/mkpoli/glyph-cutter/releases). Their provenance, licences and citation
are in [MODELS.md](MODELS.md).

## Licences

The code is MIT (`LICENSE`). The model weights are CC BY-SA 4.0 (`LICENSE-MODELS`), because they were
trained on CC BY-SA 4.0 data. Crops you cut keep the rights of the page images they come from; the
tool does not grant rights over a book you do not have.

## Development

```sh
uv sync --extra cpu --group dev
uv run pytest
uv run ruff check
```
