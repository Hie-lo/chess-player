import chess

from chess_player.analysis import analyze_position_multipv, evaluate_guess
from chess_player.engine import SimpleEngine


def test_multipv_returns_three_ranked_lines():
    board = chess.Board()
    result = analyze_position_multipv(SimpleEngine(), board, depth=2, multipv=3)
    assert result["best_move_uci"] == "e2e4"
    assert len(result["lines"]) == 3
    evals = [line["eval_cp"] for line in result["lines"]]
    assert evals == sorted(evals, reverse=True), "lines must be best-first"
    moves = [line["move_uci"] for line in result["lines"]]
    assert len(set(moves)) == 3
    for line in result["lines"]:
        assert line["pv_san"] and line["pv_uci"]
        assert 0 <= line["win_percent_white"] <= 100


def test_multipv_finds_mate_line():
    board = chess.Board(
        "r1bqk1nr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1"
    )
    result = analyze_position_multipv(SimpleEngine(), board, depth=2, multipv=3)
    assert result["lines"][0]["move_san"].startswith("Qxf7")
    assert result["lines"][0]["mate"] is not None


def test_evaluate_guess_rewards_best_and_punishes_blunder():
    # 3...? — g6 defends; Nf6?? allows Qxf7#.
    board = chess.Board(
        "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3"
    )
    good = evaluate_guess(SimpleEngine(), board, chess.Move.from_uci("g7g6"), depth=2)
    assert good["rank"] == 1 and good["is_engine_best"] and good["accuracy"] == 100.0

    bad = evaluate_guess(SimpleEngine(), board, chess.Move.from_uci("g8f6"), depth=2)
    assert bad["rank"] is None
    assert bad["classification"] in ("mistake", "blunder")
    assert bad["accuracy"] < 50
    assert bad["best_move_san"] == "g6"
