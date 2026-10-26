import zipfile

import pytest

from chess_player.cli import build_parser
from chess_player.installer import ReleaseAsset, safe_extract, select_assets

_ASSETS = [
    ReleaseAsset("stockfish-android-arm64-universal.tar.gz", "https://x/android"),
    ReleaseAsset("stockfish-linux-arm64-universal.tar.gz", "https://x/linux-arm"),
    ReleaseAsset("stockfish-linux-x86-64-universal.tar.gz", "https://x/linux-x64"),
    ReleaseAsset("stockfish-macos-universal.tar.gz", "https://x/macos"),
    ReleaseAsset("stockfish-windows-arm64-universal.zip", "https://x/win-arm"),
    ReleaseAsset("stockfish-windows-x86-64-universal.zip", "https://x/win-x64"),
]


def test_select_assets_windows_amd64():
    selected = select_assets(_ASSETS, system="Windows", machine="AMD64")
    assert selected[0].name == "stockfish-windows-x86-64-universal.zip"


def test_select_assets_linux_and_macos():
    linux = select_assets(_ASSETS, system="Linux", machine="x86_64")
    assert linux[0].name.startswith("stockfish-linux-x86-64")
    assert all("android" not in a.name for a in linux)
    mac = select_assets(_ASSETS, system="Darwin", machine="arm64")
    assert mac[0].name.startswith("stockfish-macos")
    win_arm = select_assets(_ASSETS, system="Windows", machine="ARM64")
    assert win_arm[0].name == "stockfish-windows-arm64-universal.zip"


def test_safe_extract_zip(tmp_path):
    archive = tmp_path / "sf.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("stockfish/stockfish.exe", b"binary")
    dest = tmp_path / "out"
    safe_extract(archive, dest)
    assert (dest / "stockfish" / "stockfish.exe").read_bytes() == b"binary"


def test_safe_extract_rejects_traversal(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../evil.txt", b"x")
    with pytest.raises(ValueError):
        safe_extract(archive, tmp_path / "out")


def test_cli_has_install_command():
    args = build_parser().parse_args(["install-stockfish", "--dest", "engines"])
    assert args.dest == "engines"
    assert callable(args.func)
