"""The `glyph-cutter` command line."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import typer

from . import crop, models, segment, transcript
from .classify import Classifier
from .cut import Cutter
from .detect import Detector
from .schema import Box, Detection, Piece, Placement
from .sources import Page, iter_pages, parse_pages

app = typer.Typer(
    help="Cut square character crops out of scanned classical books.",
    no_args_is_help=True,
    add_completion=False,
)

VERSION = "0.1.0"


def _providers(device: str) -> list[str] | None:
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return None


def _model_directory(directory: Path | None, fetch: bool) -> Path:
    try:
        return models.ensure(
            directory,
            download=fetch,
            progress=models.progress_to_stderr if fetch else None,
        )
    except (FileNotFoundError, RuntimeError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error


@app.command("fetch")
def fetch_models(
    models_dir: Annotated[
        Path | None, typer.Option("--models", help="directory to put the model files in")
    ] = None,
    force: Annotated[bool, typer.Option(help="download every file again")] = False,
) -> None:
    """Download the detector, classifier and cutter models."""
    target = (models_dir or models.default_directory()).expanduser()
    try:
        models.ensure(target, download=True, force=force, progress=models.progress_to_stderr)
    except RuntimeError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    for name, record in models.FILES.items():
        typer.echo(f"{name:<18} {record['bytes']:>12,} bytes  {record['sha256'][:12]}")
    typer.echo(f"models in {target}")


@app.command("cut")
def cut_book(
    inputs: Annotated[list[Path], typer.Argument(help="PDF, image, or directory of images")],
    out: Annotated[Path, typer.Option(help="directory for crops, manifest.jsonl and run.json")] = Path("out"),
    mode: Annotated[
        str, typer.Option(help="page: cut a page with no transcription; lines: cut supplied lines")
    ] = "page",
    lines: Annotated[Path | None, typer.Option(help="JSON or JSONL line boxes for --mode lines")] = None,
    models_dir: Annotated[
        Path | None, typer.Option("--models", help="directory holding the model files")
    ] = None,
    fetch: Annotated[bool, typer.Option("--fetch", help="download missing model files")] = False,
    dpi: Annotated[int, typer.Option(min=72, max=1200, help="PDF rendering resolution")] = 300,
    pages: Annotated[str | None, typer.Option(help="pages to read, e.g. 1-5,8")] = None,
    max_pages: Annotated[int | None, typer.Option(help="stop after this many pages")] = None,
    reading: Annotated[str, typer.Option(help="rtl, ltr or none: the order crops are numbered in")] = "rtl",
    size: Annotated[
        int, typer.Option(min=0, max=4096, help="side of an output crop in pixels; 0 keeps the source square")
    ] = 384,
    padding: Annotated[
        float, typer.Option(min=0.0, max=1.0, help="paper added to each side, as a share of the box")
    ] = 0.08,
    format: Annotated[str, typer.Option(help="png, jpeg or webp")] = "png",
    quality: Annotated[int, typer.Option(min=1, max=100, help="JPEG or WebP quality")] = 92,
    score: Annotated[float, typer.Option(min=0.0, max=1.0, help="lowest detector score kept")] = 0.02,
    nms: Annotated[
        float, typer.Option(min=0.0, max=1.0, help="IoU above which a duplicate detection is dropped")
    ] = 0.2,
    device: Annotated[str, typer.Option(help="auto, cpu or cuda")] = "auto",
    sized: Annotated[
        bool, typer.Option("--sized/--no-sized", help="enlarge a page of small writing before detecting")
    ] = True,
    accepted_only: Annotated[
        bool, typer.Option("--accepted/--all", help="lines mode: keep only accepted placements")
    ] = True,
) -> None:
    """Cut characters from the pages of a book into square crops."""
    if mode not in ("page", "lines"):
        raise typer.BadParameter(f"mode must be page or lines, got {mode!r}")
    if mode == "lines" and lines is None:
        raise typer.BadParameter("--mode lines needs --lines")
    if reading not in ("rtl", "ltr", "none"):
        raise typer.BadParameter(f"reading must be rtl, ltr or none, got {reading!r}")
    if format.lower() not in ("png", "jpeg", "jpg", "webp"):
        raise typer.BadParameter(f"format must be png, jpeg or webp, got {format!r}")
    if device not in ("auto", "cpu", "cuda"):
        raise typer.BadParameter(f"device must be auto, cpu or cuda, got {device!r}")
    try:
        selected = parse_pages(pages)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    directory = _model_directory(models_dir, fetch)
    records = transcript.load(lines) if lines is not None else []
    providers = _providers(device)
    started = time.time()
    typer.echo(f"models     {directory}")
    detector = Detector(directory / "detector.onnx", score=score, nms=nms, providers=providers)
    classifier = Classifier(directory / "classifier.onnx", providers=providers)
    cutter = Cutter(directory)
    typer.echo(f"detector   {', '.join(detector.providers())}")
    typer.echo(f"classifier {', '.join(classifier.providers())}")
    manifest = crop.Manifest(out)
    crop_count = 0
    page_count = 0
    skipped = 0
    try:
        for page in iter_pages(inputs, dpi=dpi, pages=selected, max_pages=max_pages):
            page_count += 1
            image = page.image
            found = detector.boxes_sized(image) if sized else detector.boxes(image)
            detections = [Detection(box, value) for box, value in found]
            paper = crop.paper_colour(image)
            page_dir = out / "crops" / page.source_id / f"p{page.number:04d}"
            if mode == "page":
                if cutter.framing is None:
                    raise RuntimeError(f"no frame.json in {directory}")
                pieces = segment.ordered(
                    segment.cut_page(detections, image, classifier, cutter.framing), reading
                )
                if not pieces:
                    typer.echo(f"{page.source.name} p{page.number}: no characters found", err=True)
                    skipped += 1
                    continue
                for index, piece in enumerate(pieces, start=1):
                    crop_count += _write_piece(
                        manifest,
                        image,
                        paper,
                        piece,
                        page,
                        page_dir,
                        index,
                        size,
                        padding,
                        format,
                        quality,
                        out,
                    )
            else:
                page_lines = transcript.for_page(
                    records, source=page.source, source_id=page.source_id, page=page.number
                )
                if not page_lines:
                    typer.echo(f"{page.source.name} p{page.number}: no transcribed lines", err=True)
                    skipped += 1
                    continue
                placements = cutter.cut(page_lines, detections, image, classifier)
                index = 0
                for line in page_lines:
                    for placement in placements.get(line.id, []):
                        if placement.box is None or (accepted_only and not placement.accepted):
                            continue
                        index += 1
                        crop_count += _write_placement(
                            manifest,
                            image,
                            paper,
                            placement,
                            line,
                            page,
                            page_dir,
                            index,
                            size,
                            padding,
                            format,
                            quality,
                            out,
                        )
            typer.echo(f"{page.source.name} p{page.number:<4} {page_count:>5} pages  {crop_count:>7} crops")
    finally:
        manifest.close(
            {
                "tool": "glyph-cutter",
                "version": VERSION,
                "mode": mode,
                "options": {
                    "dpi": dpi,
                    "pages": pages,
                    "reading": reading,
                    "size": size,
                    "padding": padding,
                    "format": format,
                    "quality": quality,
                    "score": score,
                    "nms": nms,
                    "sized": sized,
                    "accepted_only": accepted_only,
                },
                "models": models.describe(directory),
                "sources": [str(path) for path in inputs],
                "pages": page_count,
                "crops": crop_count,
                "skipped": skipped,
                "seconds": round(time.time() - started, 1),
            }
        )
    typer.echo(f"{crop_count} crops from {page_count} pages -> {out / 'manifest.jsonl'}")


def _common_row(page: Page, image, out: Path, path: Path, square: Box, size: tuple[int, int]) -> dict:
    return {
        "source": str(page.source),
        "source_id": page.source_id,
        "page": page.number,
        "page_size": [image.width, image.height],
        "file": str(path.relative_to(out)),
        "crop": [int(square.x), int(square.y), int(square.w)],
        "size": [size[0], size[1]],
    }


def _write_piece(
    manifest: crop.Manifest,
    image,
    paper,
    piece: Piece,
    page: Page,
    page_dir: Path,
    index: int,
    size: int,
    padding: float,
    format: str,
    quality: int,
    out: Path,
) -> int:
    square_image, square = crop.crop_square(image, piece.box, padding=padding, size=size, paper=paper)
    path = page_dir / f"{index:04d}.{crop.extension(format)}"
    crop.save_crop(square_image, path, format=format, quality=quality)
    manifest.add(
        {
            "mode": "page",
            **_common_row(page, image, out, path, square, square_image.size),
            "index": index,
            "box": [round(piece.box.x), round(piece.box.y), round(piece.box.w), round(piece.box.h)],
            "probability": round(piece.probability, 6),
            "kind": piece.kind,
            "detections": list(piece.detections),
        }
    )
    return 1


def _write_placement(
    manifest: crop.Manifest,
    image,
    paper,
    placement: Placement,
    line,
    page: Page,
    page_dir: Path,
    index: int,
    size: int,
    padding: float,
    format: str,
    quality: int,
    out: Path,
) -> int:
    square_image, square = crop.crop_square(image, placement.box, padding=padding, size=size, paper=paper)
    path = page_dir / f"{index:04d}.{crop.extension(format)}"
    crop.save_crop(square_image, path, format=format, quality=quality)
    manifest.add(
        {
            "mode": "lines",
            **_common_row(page, image, out, path, square, square_image.size),
            "index": index,
            "line": line.id,
            "seq": line.seq,
            "text": placement.token.text,
            "box": [
                round(placement.box.x),
                round(placement.box.y),
                round(placement.box.w),
                round(placement.box.h),
            ],
            "kind": placement.kind,
            "accepted": placement.accepted,
            "probability": round(placement.probability, 6),
        }
    )
    return 1


@app.command("info")
def info() -> None:
    """Print the version, the default model directory and the model checksums."""
    typer.echo(
        json.dumps(
            {"version": VERSION, "models": str(models.default_directory())}, ensure_ascii=False
        )
    )
    typer.echo(models.index())


@app.command("version")
def version() -> None:
    """Print the version."""
    typer.echo(VERSION)
