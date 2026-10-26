import chess

from chess_player.engine import SimpleEngine, get_engine


def test_simple_engine_finds_mate_in_one():
    # Scholar's-mate pattern: Qxf7 is mate.
    board = chess.Board("r1bqk1nr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1")
    result = SimpleEngine().analyse(board, depth=2)
    assert result.best_move is not None
    board.push(result.best_move)
    assert board.is_checkmate()


def test_simple_engine_grabs_hanging_queen():
    # Black queen on d5 is undefended and en prise to the e4 pawn: exd5 wins it.
    board = chess.Board("rnb1kbnr/ppp2ppp/8/3qp3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
    result = SimpleEngine().analyse(board, depth=2)
    assert result.best_move is not None and result.best_move.uci() == "e4d5"
    assert result.score_cp > 300, "engine should see it is winning material"


def test_terminal_checkmate_score():
    board = chess.Board()
    for san in ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7#"]:
        board.push_san(san)
    result = SimpleEngine().analyse(board, depth=2)
    assert result.best_move is None
    assert result.score_cp < -9000  # side to move (Black) is mated


def test_get_engine_simple():
    engine = get_engine("simple")
    assert "SimpleEngine" in engine.name
    engine.close()
