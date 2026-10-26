import chess
import pytest

pytest.importorskip("PIL")
pytest.importorskip("numpy")

from chess_player.image_fen import (  # noqa: E402
    board_grid_from_image,
    fen_from_grid,
    render_board_image,
)


def _expected_grid(board):
    grid = []
    for rank in range(7, -1, -1):
        row = []
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            row.append(piece.symbol() if piece else "")
        grid.append(row)
    return grid


@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1",
        "4kb1r/p2r1ppp/4qn2/1B2p1B1/4P3/1Q6/PPP2PPP/2KR4 w k - 0 1",
    ],
)
def test_round_trip_render_then_recognise(fen):
    """Render a clean digital board, then read it back.

    Real-world screenshots (other piece sets/themes) will be less
    accurate — that is why the UI has a correction editor — but the
    pipeline itself must be exact on the style it was templated from.
    """
    board = chess.Board(fen)
    proposal = board_grid_from_image(render_board_image(board))
    assert proposal["grid"] == _expected_grid(board)
    rebuilt = fen_from_grid(proposal["grid"], "white")
    assert chess.Board(rebuilt).board_fen() == board.board_fen()
    assert len(proposal["confidences"]) == 8
