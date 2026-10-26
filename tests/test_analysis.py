import io

import chess
import chess.pgn

from chess_player.analysis import analyze_game, classify, move_accuracy, win_percent
from chess_player.engine import SimpleEngine
from chess_player.report import to_html, to_json, to_markdown

BLUNDER_PGN = """[Event "test"]
[White "A"]
[Black "B"]
[Result "1-0"]

1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0
"""


def test_win_percent_bounds():
    assert win_percent(0) == 50.0
    assert 50 < win_percent(200) < 100
    assert 0 < win_percent(-200) < 50
    assert win_percent(100_000) < 99.9  # clipped, no overflow


def test_move_accuracy_monotonic():
    assert move_accuracy(50, 50) == 100.0
    assert move_accuracy(50, 60) == 100.0  # improvement is not penalised
    assert move_accuracy(50, 40) < move_accuracy(50, 45)
    assert move_accuracy(90, 10) < 5  # catastrophic move


def test_classify_thresholds():
    assert classify(True, 99) == "best"
    assert classify(False, 1) == "excellent"
    assert classify(False, 4) == "good"
    assert classify(False, 8) == "inaccuracy"
    assert classify(False, 15) == "mistake"
    assert classify(False, 50) == "blunder"


def test_analyze_game_flags_the_blunder():
    game = chess.pgn.read_game(io.StringIO(BLUNDER_PGN))
    analysis = analyze_game(SimpleEngine(), game, depth=2)
    assert len(analysis.moves) == 7
    # 3...Nf6?? walks into Qxf7# — even the weak fallback engine must flag it.
    nf6 = analysis.moves[5]
    assert nf6.san == "Nf6"
    assert nf6.classification in ("mistake", "blunder")
    assert nf6.win_loss > 10
    # Aggregate numbers are present and sane.
    assert analysis.accuracy("black") is not None
    assert analysis.accuracy("black") < 90
    d = analysis.to_dict()
    assert d["moves"][5]["best_move_san"] != "Nf6"


def test_reports_render():
    game = chess.pgn.read_game(io.StringIO(BLUNDER_PGN))
    analysis = analyze_game(SimpleEngine(), game, depth=2)
    assert "Qxf7" in to_markdown(analysis) or "Nf6" in to_markdown(analysis)
    assert '"white_accuracy"' in to_json(analysis)
    assert "<svg" in to_html(analysis)
