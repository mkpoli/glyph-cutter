"""The model files the cutter reads, their checksums, and their download."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path

#: The release that carries the weights.
VERSION = "0.1.0"
RELEASE = f"https://github.com/mkpoli/glyph-cutter/releases/download/v{VERSION}"

#: Every file a cut needs, with the SHA-256 it must have.
FILES: dict[str, dict[str, str | int]] = {
    "detector.onnx": {
        "sha256": "ac21e5db5ec1118ce1e812d73f6daa7aeddc501df5c01aa4d27f5a7ba045a752",
        "bytes": 80845313,
    },
    "classifier.onnx": {
        "sha256": "43601b024ca070dfbae5d71844ebc6fe05267191da56e15e6e2c46d1d50800f6",
        "bytes": 125551320,
    },
    "classes.json": {
        "sha256": "f9b35c02863126b116affa28b3eb2a9459ada3e0851a017f1871ea0fff770920",
        "bytes": 41170,
    },
    "frame.json": {
        "sha256": "3772fb14ec0c46aae91b3b0dce9940a0fe1a93738b9af73b6f8902a35f44e6b0",
        "bytes": 1594,
    },
    "gate.json": {
        "sha256": "0ed9433a24d56cb1804347ba4fbfd5ee21e18aa724b2bf48941a5fde2bad2968",
        "bytes": 3470,
    },
}


def default_directory() -> Path:
    """Where the weights live: `$GLYPH_CUTTER_MODELS`, or the XDG cache."""
    override = os.environ.get("GLYPH_CUTTER_MODELS")
    if override:
        return Path(override).expanduser()
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "glyph-cutter" / "models"


def missing(directory: Path) -> list[str]:
    """The model files absent from `directory`."""
    return [name for name in FILES if not (directory / name).is_file()]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def verified(directory: Path) -> list[str]:
    """The model files present but not matching their recorded checksum."""
    bad = []
    for name, record in FILES.items():
        path = directory / name
        if path.is_file() and digest(path) != record["sha256"]:
            bad.append(name)
    return bad


def _download(name: str, directory: Path, progress: Callable[[str, int, int], None] | None) -> Path:
    url = f"{RELEASE}/{name}"
    target = directory / name
    partial = directory / f"{name}.part"
    directory.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": f"glyph-cutter/{VERSION}"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        read = 0
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            handle.write(block)
            read += len(block)
            if progress is not None:
                progress(name, read, total)
    if digest(partial) != FILES[name]["sha256"]:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"{url}: the download does not match its SHA-256")
    partial.replace(target)
    return target


def ensure(
    directory: Path | None = None,
    *,
    download: bool = False,
    force: bool = False,
    progress: Callable[[str, int, int], None] | None = None,
) -> Path:
    """The model directory, downloading every missing file when `download` is true."""
    directory = (directory or default_directory()).expanduser()
    if force:
        for name in FILES:
            (directory / name).unlink(missing_ok=True)
    absent = missing(directory)
    if absent and download:
        for name in absent:
            _download(name, directory, progress)
        absent = missing(directory)
    if absent:
        raise FileNotFoundError(
            f"missing {', '.join(absent)} in {directory}; run `glyph-cutter fetch`"
        )
    bad = verified(directory)
    if bad:
        raise RuntimeError(
            f"{', '.join(bad)} in {directory} does not match its SHA-256; "
            "run `glyph-cutter fetch --force`"
        )
    return directory


def describe(directory: Path) -> dict:
    """What a run reads: every model file, its size and its checksum."""
    return {
        name: {
            "bytes": (directory / name).stat().st_size,
            "sha256": FILES[name]["sha256"],
        }
        for name in FILES
        if (directory / name).is_file()
    }


def index() -> str:
    """The model index as JSON, for a release or a bug report."""
    return json.dumps({"release": RELEASE, "files": FILES}, indent=2)


def progress_to_stderr() -> Callable[[str, int, int], None]:
    """A simple progress reporter for the download."""

    def report(name: str, read: int, total: int) -> None:
        if total:
            share = read / total
            print(
                f"\r{name:<18} {share:6.1%} {read / 1e6:6.1f}/{total / 1e6:6.1f} MB",
                end="",
                file=sys.stderr,
            )
        else:
            print(f"\r{name:<18} {read / 1e6:6.1f} MB", end="", file=sys.stderr)
        if read >= total:
            print(file=sys.stderr)

    return report
