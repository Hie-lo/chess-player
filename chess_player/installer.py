"""Download and install the real Stockfish binary for this project.

Why this exists: ``pip install stockfish`` installs only a small Python
*wrapper* (a few ``.py`` files) — it contains no engine.  The engine is a
native executable that must be downloaded from the Stockfish project.
This module does that in one command::

    python -m chess_player install-stockfish

It asks the GitHub API for the latest official Stockfish release, picks
the asset for the current OS/CPU, downloads it into ``engines/``,
extracts it, launches every candidate it finds until one answers UCI
and analyses a test position, and leaves that binary where the normal
engine discovery (:func:`chess_player.engine.stockfish_candidates`)
will find it from then on.
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

_RELEASES_API = "https://api.github.com/repos/official-stockfish/Stockfish/releases/latest"
_FALLBACK_TAG = "sf_19"
_FALLBACK_ASSET_NAMES = [
    "stockfish-windows-x86-64-universal.zip",
    "stockfish-windows-arm64-universal.zip",
    "stockfish-linux-x86-64-universal.tar.gz",
    "stockfish-linux-arm64-universal.tar.gz",
    "stockfish-linux-riscv64-universal.tar.gz",
    "stockfish-macos-universal.tar.gz",
]
_USER_AGENT = "chess-player-installer/0.1 (+https://github.com/Hie-lo/chess-player)"


@dataclass(frozen=True)
class ReleaseAsset:
    name: str
    url: str
    size: int = 0


def _fallback_assets() -> list[ReleaseAsset]:
    base = (
        "https://github.com/official-stockfish/Stockfish/releases/download/"
        f"{_FALLBACK_TAG}/"
    )
    return [ReleaseAsset(name, base + name) for name in _FALLBACK_ASSET_NAMES]


def latest_release_assets(timeout: float = 30) -> tuple[str, list[ReleaseAsset]]:
    """Latest official release's assets via the GitHub API.

    Falls back to a pinned known-good release if the API is unreachable
    (offline, rate-limited, …) so the printed manual URL stays useful.
    """
    request = urllib.request.Request(
        _RELEASES_API,
        headers={"Accept": "application/vnd.github+json", "User-Agent": _USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
        assets = [
            ReleaseAsset(a["name"], a["browser_download_url"], int(a.get("size", 0)))
            for a in data.get("assets", [])
        ]
        if assets:
            return str(data.get("tag_name") or _FALLBACK_TAG), assets
    except Exception:
        pass
    return _FALLBACK_TAG, _fallback_assets()


def select_assets(
    assets: list[ReleaseAsset],
    *,
    system: str | None = None,
    machine: str | None = None,
) -> list[ReleaseAsset]:
    """Assets for this OS/CPU, most suitable first (empty = unsupported)."""
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()

    def name_of(asset: ReleaseAsset) -> str:
        return asset.name.lower()

    # Exact OS+arch matches only — downloading an ARM build on x86 (or
    # vice versa) just wastes ~80 MB.  Fall back to any build for the OS
    # only when no asset names this architecture at all (e.g. a single
    # "universal" macOS build).
    if system.startswith("win"):
        exact = "windows-arm64" if machine in ("arm64", "aarch64") else "windows-x86-64"
        ordered = [a for a in assets if exact in name_of(a)] or [
            a for a in assets if "windows" in name_of(a)
        ]
    elif system == "darwin":
        ordered = [a for a in assets if "macos" in name_of(a)]
    elif system.startswith("linux"):
        if machine in ("aarch64", "arm64"):
            exact = "linux-arm64"
        elif machine.startswith("riscv"):
            exact = "linux-riscv64"
        else:
            exact = "linux-x86-64"
        ordered = [
            a for a in assets if exact in name_of(a) and "android" not in name_of(a)
        ] or [
            a for a in assets if "linux" in name_of(a) and "android" not in name_of(a)
        ]
    else:
        ordered = []

    # Deduplicate while keeping order; prefer "universal"/plain builds,
    # then common compatible variants, AVX-512 last (see _variant_rank).
    from .engine import _variant_rank

    seen: set[str] = set()
    unique: list[ReleaseAsset] = []
    for asset in ordered:
        if asset.name not in seen:
            seen.add(asset.name)
            unique.append(asset)
    unique.sort(key=lambda a: _variant_rank(a.name.replace("universal", "")))
    return unique


def manual_download_url() -> str | None:
    """Best URL to show the user if the automated download fails."""
    _, assets = latest_release_assets(timeout=10)
    selected = select_assets(assets)
    return selected[0].url if selected else None


def download_file(url: str, dest: Path, *, timeout: float = 60) -> Path:
    """Stream a URL to disk with progress (Release assets are ~80 MB)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    print(f"⬇️ در حال دانلود: {url}", flush=True)
    with urllib.request.urlopen(request, timeout=timeout) as response, open(
        dest, "wb"
    ) as fh:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1024 * 1024):
            fh.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r   {done / 1e6:6.1f} / {total / 1e6:6.1f} MB "
                      f"({done * 100 // total}٪)", end="", flush=True)
    print(flush=True)
    return dest


def _is_within(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def safe_extract(archive: Path, dest: Path) -> None:
    """Extract .zip/.tar(.gz) refusing path-traversal members."""
    dest.mkdir(parents=True, exist_ok=True)
    lower = archive.name.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                if not _is_within(dest, dest / member):
                    raise ValueError(f"عضو ناامن در فایل فشرده: {member}")
            zf.extractall(dest)
    elif lower.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                if not _is_within(dest, dest / member.name):
                    raise ValueError(f"عضو ناامن در فایل فشرده: {member.name}")
            tf.extractall(dest)
    else:
        raise ValueError(f"قالب فایل فشرده پشتیبانی نمی‌شود: {archive.name}")


def _test_binary(path: str) -> str:
    """Launch a binary and analyse one position; return its UCI name."""
    import chess

    from .engine import _try_stockfish

    engine = _try_stockfish(path)
    try:
        board = chess.Board()
        result = engine.analyse(board, depth=6)
        if result.best_move is None:  # pragma: no cover - defensive
            raise RuntimeError("موتور هیچ حرکتی برنگرداند")
        return engine.name
    finally:
        engine.close()


def _working_stockfish() -> str | None:
    from .engine import stockfish_candidates

    for candidate in stockfish_candidates():
        try:
            _test_binary(candidate)
            return candidate
        except Exception:
            continue
    return None


def pip_wrapper_version() -> str | None:
    """Version of the PyPI ``stockfish`` *wrapper* package, if installed.

    That package is only Python glue (no engine binary inside), which is
    a very common source of confusion — doctor calls this out by name.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return None
    try:
        return version("stockfish")
    except PackageNotFoundError:
        return None


def install_stockfish(
    dest_dir: Path = Path("engines"), *, force: bool = False
) -> Path:
    """Install a working Stockfish under ``dest_dir``; return its path."""
    if not force:
        existing = _working_stockfish()
        if existing:
            print(f"✓ Stockfish همین حالا هم کار می‌کند: {existing}", flush=True)
            return Path(existing)

    tag, assets = latest_release_assets()
    candidates = select_assets(assets)
    if not candidates:
        raise RuntimeError(
            f"برای سیستم {platform.system()} {platform.machine()} فایل آماده‌ای "
            "در انتشار رسمی پیدا نشد؛ Stockfish را دستی از stockfishchess.org "
            "بگیرید و مسیرش را با --engine بدهید."
        )
    print(f"انتشار رسمی: {tag} — {len(candidates)} گزینه برای سیستم شما", flush=True)

    dest_dir = Path(dest_dir)
    download_dir = dest_dir / "_downloads"
    last_error: Exception | None = None
    for asset in candidates:
        archive = download_dir / asset.name
        try:
            download_file(asset.url, archive)
            print("📦 در حال خارج‌کردن از فشرده‌سازی…", flush=True)
            safe_extract(archive, dest_dir)
        except (OSError, urllib.error.URLError, ValueError) as exc:
            last_error = exc
            print(f"⚠️ دانلود/بازکردن {asset.name} ناموفق بود: {exc}", flush=True)
            continue

        from .engine import stockfish_candidates

        found = [
            c for c in stockfish_candidates(roots=[dest_dir])
            if str(dest_dir) in c or Path(c).is_relative_to(dest_dir)
        ]
        for candidate in found:
            try:
                name = _test_binary(candidate)
            except Exception as exc:
                last_error = exc
                print(f"   ✗ {candidate}: {type(exc).__name__}: {exc}", flush=True)
                continue
            print(f"✓ موتور آماده است: {name}\n✓ مسیر: {candidate}", flush=True)
            # Tidy up the (large) archive; the extracted engine stays.
            shutil.rmtree(download_dir, ignore_errors=True)
            return Path(candidate)
        print("⚠️ در این بیلد فایل اجراییِ کارکننده‌ای پیدا نشد؛ گزینهٔ بعدی…", flush=True)

    raise RuntimeError(
        "نصب خودکار کامل نشد"
        + (f" (آخرین خطا: {type(last_error).__name__}: {last_error})" if last_error else "")
    )


if __name__ == "__main__":  # pragma: no cover - manual entry point
    try:
        installed = install_stockfish()
    except Exception as exc:
        print(f"✗ {exc}", file=sys.stderr)
        url = manual_download_url()
        if url:
            print(f"دانلود دستی از مرورگر: {url}", file=sys.stderr)
        raise SystemExit(1)
    print(f"حالا اجرا کنید:  python -m chess_player serve --port 8000\n({installed})")
