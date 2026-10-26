from pathlib import Path

from chess_player.cli import main
from chess_player.engine import stockfish_candidates


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-binary")
    return path


def test_candidates_find_binary_dropped_in_project(tmp_path, monkeypatch):
    monkeypatch.delenv("STOCKFISH_PATH", raising=False)
    in_engines = _touch(tmp_path / "engines" / "stockfish-windows-x86-64-avx2.exe")
    in_folder = _touch(tmp_path / "stockfish" / "stockfish.exe")
    # Noise that must NOT be treated as a binary:
    _touch(tmp_path / "stockfish.zip")  # un-extracted download
    _touch(tmp_path / "stockfish.py")
    _touch(tmp_path / "notes" / "stockfish.md")

    candidates = stockfish_candidates(roots=[tmp_path])
    local = [c for c in candidates if str(tmp_path) in c]
    assert str(in_folder) in local
    assert str(in_engines) in local
    assert not any(c.endswith((".zip", ".py", ".md")) for c in local)


def test_candidates_prefer_plain_and_safe_builds(tmp_path):
    avx512 = _touch(tmp_path / "stockfish" / "stockfish-windows-x86-64-avx512.exe")
    plain = _touch(tmp_path / "stockfish.exe")
    candidates = [
        c for c in stockfish_candidates(roots=[tmp_path]) if str(tmp_path) in c
    ]
    assert candidates.index(str(plain)) < candidates.index(str(avx512))


def test_doctor_with_simple_engine(capsys):
    assert main(["doctor", "--engine", "simple"]) == 0
    out = capsys.readouterr().out
    assert "doctor" in out
    assert "SimpleEngine" in out
    assert "تست تحلیل" in out


def test_doctor_reports_missing_stockfish(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # empty folder: nothing to find locally
    monkeypatch.delenv("STOCKFISH_PATH", raising=False)
    # Exit code depends on whether the host machine has Stockfish on PATH;
    # either way it must produce a diagnostic report, not crash.
    code = main(["doctor"])
    out = capsys.readouterr().out
    assert code in (0, 2)
    assert "نامزدهای Stockfish" in out
