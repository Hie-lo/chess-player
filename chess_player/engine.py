"""Engine layer: Stockfish over UCI, plus a small built-in fallback engine.

Why two engines?
----------------
Stockfish is the real analysis engine, but it is a native binary that users
install separately (``apt install stockfish`` / ``brew install stockfish`` /
download from stockfishchess.org).  So that the tool, its tests and the demo
web app still run on a machine without Stockfish, :class:`SimpleEngine`
provides a weak-but-honest negamax search (alpha-beta, material + simple
piece-square terms).  It is good enough to spot hanging pieces and mates in
1-2, which is all the tests need — for serious numbers, use Stockfish.
Both engines share one interface, :meth:`analyse`.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field

import chess
import chess.engine

MATE_SCORE = 100_000  # internal "mate" value used by SimpleEngine
#: Centipawn value assigned to a forced mate when reporting scores.
#: (Win%/accuracy maths clips to +-1000 cp anyway; see analysis.py.)
MATE_CP = 10_000


@dataclass
class EngineResult:
    """Result of analysing one position.

    ``score_cp`` is always from the point of view of the side to move in
    the analysed position (positive = side to move is better).  A forced
    mate is reported as ``mate`` (moves to mate, signed from the side to
    move's POV) and mirrored into ``score_cp`` as +-MATE_CP.
    """

    best_move: chess.Move | None
    score_cp: int
    mate: int | None = None
    pv: list[chess.Move] = field(default_factory=list)
    depth: int = 0
    engine_name: str = ""

    @property
    def best_move_uci(self) -> str | None:
        return self.best_move.uci() if self.best_move else None


def _terminal_result(board: chess.Board, engine_name: str) -> EngineResult | None:
    """Return a result for mate/stalemate/draw positions, else None."""
    if board.is_checkmate():
        # Side to move has been mated.
        return EngineResult(None, -MATE_CP, mate=0, engine_name=engine_name)
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return EngineResult(None, 0, mate=None, engine_name=engine_name)
    return None


# --------------------------------------------------------------------------
# Stockfish (UCI)
# --------------------------------------------------------------------------


class StockfishEngine:
    """Stockfish driven through the UCI protocol via python-chess.

    UCI refresher (what python-chess sends under the hood)::

        uci
        setoption name Threads value 2
        position fen <FEN>
        go depth 14
        info depth 14 score cp 37 pv e2e4 e7e5 ...
        bestmove e2e4
    """

    name = "Stockfish (UCI)"

    def __init__(self, path: str, *, threads: int = 2, hash_mb: int = 128) -> None:
        self._engine = chess.engine.SimpleEngine.popen_uci(path)
        options: dict[str, int] = {}
        if "Threads" in self._engine.options:
            options["Threads"] = threads
        if "Hash" in self._engine.options:
            options["Hash"] = hash_mb
        if options:
            self._engine.configure(options)
        protocol_name = getattr(self._engine.protocol, "id", {}).get("name")
        if protocol_name:
            self.name = protocol_name

    def analyse(
        self,
        board: chess.Board,
        *,
        depth: int = 14,
        time_limit: float | None = None,
    ) -> EngineResult:
        terminal = _terminal_result(board, self.name)
        if terminal is not None:
            return terminal
        limit = (
            chess.engine.Limit(time=time_limit)
            if time_limit is not None
            else chess.engine.Limit(depth=depth)
        )
        info = self._engine.analyse(board, limit)
        score = info["score"].pov(board.turn)
        pv = list(info.get("pv", []))
        return EngineResult(
            best_move=pv[0] if pv else None,
            score_cp=int(score.score(mate_score=MATE_CP)),
            mate=score.mate(),
            pv=pv,
            depth=int(info.get("depth", depth)),
            engine_name=self.name,
        )

    def close(self) -> None:
        try:
            self._engine.quit()
        except Exception:
            pass

    def __enter__(self) -> "StockfishEngine":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --------------------------------------------------------------------------
# Fallback engine (didactic, weak — NOT a substitute for Stockfish)
# --------------------------------------------------------------------------

_PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Tiny piece-square nudges, from White's perspective (a1 = index 0).
_PAWN_PST = [0] * 64
for _sq in range(64):
    _rank, _file = chess.square_rank(_sq), chess.square_file(_sq)
    _centre = 3.5 - abs(3.5 - _file)  # 3.5 on e/d files, 0.5 on a/h files
    # Reward advanced pawns, central pawns most of all (so 1.e4 > 1.h4).
    _PAWN_PST[_sq] = int(_rank * (1 + 5 * _centre))

_KNIGHT_PST = [0] * 64
for _sq in range(64):
    _file, _rank = chess.square_file(_sq), chess.square_rank(_sq)
    _KNIGHT_PST[_sq] = int((3.5 - abs(3.5 - _file)) * 4 + (3.5 - abs(3.5 - _rank)) * 2)


def _evaluate(board: chess.Board) -> int:
    """Static eval in centipawns, from the side-to-move's perspective."""
    if board.is_checkmate():
        return -MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    score = 0  # from White's perspective first
    for square, piece in board.piece_map().items():
        value = _PIECE_VALUES[piece.piece_type]
        if piece.piece_type == chess.PAWN:
            value += _PAWN_PST[square if piece.color == chess.WHITE else chess.square_mirror(square)]
        elif piece.piece_type == chess.KNIGHT:
            value += _KNIGHT_PST[square if piece.color == chess.WHITE else chess.square_mirror(square)]
        score += value if piece.color == chess.WHITE else -value
    return score if board.turn == chess.WHITE else -score


def _ordered_moves(board: chess.Board) -> list[chess.Move]:
    """Captures first (MVV-LVA-ish) so alpha-beta prunes much better."""

    def key(move: chess.Move) -> int:
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        score = 0
        if victim is not None:
            score = 10 * _PIECE_VALUES[victim.piece_type] - (
                _PIECE_VALUES[attacker.piece_type] // 100 if attacker else 0
            )
        if move.promotion:
            score += _PIECE_VALUES[move.promotion]
        return -score

    return sorted(board.legal_moves, key=key)


class SimpleEngine:
    """Negamax + alpha-beta. Tries mates and hanging pieces; nothing more."""

    name = "SimpleEngine (built-in fallback, weak — install Stockfish for real analysis)"

    def __init__(self, max_depth: int = 3) -> None:
        self.max_depth = max_depth

    def analyse(
        self,
        board: chess.Board,
        *,
        depth: int = 2,
        time_limit: float | None = None,  # accepted for API compatibility; ignored
    ) -> EngineResult:
        terminal = _terminal_result(board, self.name)
        if terminal is not None:
            return terminal
        depth = max(1, min(depth, self.max_depth))
        score, pv = self._negamax(board, depth, -MATE_SCORE - 1, MATE_SCORE + 1, 0)
        mate = None
        reported_cp = score
        if abs(score) > MATE_SCORE - 1000:
            plies = MATE_SCORE - abs(score)
            moves_to_mate = (plies + 1) // 2
            mate = moves_to_mate if score > 0 else -moves_to_mate
            reported_cp = MATE_CP if score > 0 else -MATE_CP
        return EngineResult(
            best_move=pv[0] if pv else None,
            score_cp=int(reported_cp),
            mate=mate,
            pv=pv,
            depth=depth,
            engine_name=self.name,
        )

    def _quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        """Captures-only search at the horizon.

        Without this, depth-2 search "wins" a pawn with Nxe5 and never sees
        the recapture one ply later (the horizon effect).  Following only
        captures until the position is quiet makes the weak fallback engine
        dramatically less silly at negligible cost.
        """
        stand_pat = _evaluate(board)
        if stand_pat > MATE_SCORE - 1000:
            return stand_pat - ply  # faster mate is better
        if stand_pat < -MATE_SCORE + 1000:
            return stand_pat + ply
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)
        for move in _ordered_moves(board):
            if not board.is_capture(move):
                continue
            board.push(move)
            score = -self._quiescence(board, -beta, -alpha, ply + 1)
            board.pop()
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha

    def _negamax(
        self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int
    ) -> tuple[int, list[chess.Move]]:
        if depth == 0:
            return self._quiescence(board, alpha, beta, ply), []
        if board.is_game_over():
            score = _evaluate(board)
            if score > MATE_SCORE - 1000:
                score -= ply
            elif score < -MATE_SCORE + 1000:
                score += ply
            return score, []

        best_score, best_pv = -MATE_SCORE - 1, []
        for move in _ordered_moves(board):
            board.push(move)
            child_score, child_pv = self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            score = -child_score
            if score > best_score:
                best_score, best_pv = score, [move, *child_pv]
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best_score, best_pv

    def close(self) -> None:  # symmetry with StockfishEngine
        pass

    def __enter__(self) -> "SimpleEngine":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def find_stockfish(explicit_path: str | None = None) -> str | None:
    """Locate a Stockfish binary: explicit path, $STOCKFISH_PATH, then PATH."""
    import os

    for candidate in (explicit_path, os.environ.get("STOCKFISH_PATH")):
        if candidate:
            return candidate
    return shutil.which("stockfish")


def get_engine(engine_arg: str | None = "auto"):
    """Create an engine.

    ``engine_arg``:
      * ``"auto"`` (default) — Stockfish if found, else SimpleEngine;
      * ``"simple"`` — always the fallback engine;
      * any other string — treated as a path to a UCI engine binary.
    """
    if engine_arg == "simple":
        return SimpleEngine()
    if engine_arg in (None, "auto"):
        path = find_stockfish()
        return StockfishEngine(path) if path else SimpleEngine()
    return StockfishEngine(engine_arg)
