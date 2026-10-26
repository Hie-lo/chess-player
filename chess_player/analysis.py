"""Move-quality analysis: centipawn loss, Win%, accuracy, classification.

The maths follows the (open) Lichess model, which Chess.com's Game Review
implements similarly:

* Engine scores are in centipawns (cp), clipped to +-1000 for Win% purposes.
* Win%  = 50 + 50 * (2 / (1 + exp(-0.00368208 * cp)) - 1)
* Per-move accuracy from the Win% thrown away by the move:
      Accuracy = 103.1668 * exp(-0.04354 * (win_before - win_after)) - 3.1669
  clipped to [0, 100]; a move that *improves* your Win% scores 100.
* Classification is by Win% lost (more meaningful than raw cp near mate
  or in already-decided positions)::

      engine's own move        -> Best
      win% lost <=  2          -> Excellent
      win% lost <=  5          -> Good
      win% lost <= 10          -> Inaccuracy  (?!)
      win% lost <= 20          -> Mistake     (?)
      otherwise                -> Blunder     (??)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import chess

from .engine import EngineResult

WIN_PERCENT_CP_CAP = 1000


def win_percent(cp: int) -> float:
    """Expected score (0-100) for the side the cp value favours."""
    cp = max(-WIN_PERCENT_CP_CAP, min(WIN_PERCENT_CP_CAP, cp))
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def move_accuracy(win_before: float, win_after: float) -> float:
    """Accuracy (0-100) of a move given Win% before/after, mover's POV."""
    loss = win_before - win_after
    if loss <= 0:
        return 100.0
    return max(0.0, min(100.0, 103.1668 * math.exp(-0.04354 * loss) - 3.1669))


LABELS_FA = {
    "best": "بهترین",
    "excellent": "عالی",
    "good": "خوب",
    "inaccuracy": "عدم دقت ؟!",
    "mistake": "اشتباه ؟",
    "blunder": "اشتباه فاحش ؟؟",
}


def classify(is_engine_best: bool, win_loss: float) -> str:
    if is_engine_best:
        return "best"
    if win_loss <= 2:
        return "excellent"
    if win_loss <= 5:
        return "good"
    if win_loss <= 10:
        return "inaccuracy"
    if win_loss <= 20:
        return "mistake"
    return "blunder"


@dataclass
class MoveAnalysis:
    ply: int  # 0 = White's first move (1.), 1 = Black's first move (1...)
    san: str
    uci: str
    color: str  # "white" | "black"
    fen_before: str
    eval_before_cp: int  # engine score of the position, mover's POV
    eval_after_cp: int  # engine score after the played move, mover's POV
    eval_after_white_cp: int  # same, White's POV (for the eval graph)
    best_move_san: str | None
    best_move_uci: str | None
    best_line_san: list[str]
    cp_loss: int  # >= 0; how much worse the played move is than the best
    win_before: float
    win_after: float
    win_loss: float
    accuracy: float
    classification: str  # key of LABELS_FA

    @property
    def move_number(self) -> int:
        return self.ply // 2 + 1

    @property
    def label_fa(self) -> str:
        return LABELS_FA[self.classification]

    def to_dict(self) -> dict:
        return {
            "ply": self.ply,
            "move_number": self.move_number,
            "san": self.san,
            "uci": self.uci,
            "color": self.color,
            "fen_before": self.fen_before,
            "eval_before_cp": self.eval_before_cp,
            "eval_after_cp": self.eval_after_cp,
            "eval_after_white_cp": self.eval_after_white_cp,
            "best_move_san": self.best_move_san,
            "best_move_uci": self.best_move_uci,
            "best_line_san": self.best_line_san,
            "cp_loss": self.cp_loss,
            "win_before": round(self.win_before, 1),
            "win_after": round(self.win_after, 1),
            "win_loss": round(self.win_loss, 1),
            "accuracy": round(self.accuracy, 1),
            "classification": self.classification,
            "classification_fa": self.label_fa,
        }


@dataclass
class GameAnalysis:
    moves: list[MoveAnalysis]
    engine_name: str
    depth: int
    headers: dict[str, str] = field(default_factory=dict)
    result: str = "*"

    # -- aggregate stats -------------------------------------------------
    def _by_color(self, color: str) -> list[MoveAnalysis]:
        return [m for m in self.moves if m.color == color]

    def accuracy(self, color: str) -> float | None:
        moves = self._by_color(color)
        if not moves:
            return None
        return round(sum(m.accuracy for m in moves) / len(moves), 1)

    def acpl(self, color: str) -> float | None:
        """Average centipawn loss (lower is better)."""
        moves = self._by_color(color)
        if not moves:
            return None
        return round(sum(min(m.cp_loss, 1000) for m in moves) / len(moves), 1)

    def counts(self, color: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for m in self._by_color(color):
            out[m.classification] = out.get(m.classification, 0) + 1
        return out

    def to_dict(self) -> dict:
        return {
            "engine": self.engine_name,
            "depth": self.depth,
            "headers": self.headers,
            "result": self.result,
            "white_accuracy": self.accuracy("white"),
            "black_accuracy": self.accuracy("black"),
            "white_acpl": self.acpl("white"),
            "black_acpl": self.acpl("black"),
            "white_counts": self.counts("white"),
            "black_counts": self.counts("black"),
            "moves": [m.to_dict() for m in self.moves],
        }


def _pv_san(board: chess.Board, pv: list[chess.Move], limit: int = 6) -> list[str]:
    sans, tmp = [], board.copy()
    for move in pv[:limit]:
        sans.append(tmp.san(move))
        tmp.push(move)
    return sans


def analyze_position(engine, board: chess.Board, *, depth: int = 14) -> dict:
    """Best move + eval for ONE manually entered position (analysis board).

    This is the building block for studying a position *after* a game, or
    from a book/puzzle — the user supplies the FEN explicitly.
    """
    result: EngineResult = engine.analyse(board, depth=depth)
    return {
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "eval_cp": result.score_cp,  # side-to-move POV
        "mate": result.mate,
        "win_percent": round(win_percent(result.score_cp), 1),
        "best_move_uci": result.best_move_uci,
        "best_move_san": board.san(result.best_move) if result.best_move else None,
        "best_line_san": _pv_san(board, result.pv),
        "depth": result.depth,
        "engine": result.engine_name,
    }


def _line_to_dict(board: chess.Board, result: EngineResult, rank: int) -> dict:
    eval_white = result.score_cp if board.turn == chess.WHITE else -result.score_cp
    return {
        "rank": rank,
        "eval_cp": result.score_cp,  # side-to-move POV
        "eval_white_cp": eval_white,
        "mate": result.mate,  # side-to-move POV
        "win_percent": round(win_percent(result.score_cp), 1),  # for side to move
        "win_percent_white": round(win_percent(eval_white), 1),
        "move_uci": result.best_move_uci,
        "move_san": board.san(result.best_move) if result.best_move else None,
        "pv_uci": [m.uci() for m in result.pv],
        "pv_san": _pv_san(board, result.pv, limit=8),
        "depth": result.depth,
    }


def analyze_position_multipv(
    engine, board: chess.Board, *, depth: int = 14, multipv: int = 3
) -> dict:
    """Live analysis of one position: top-N candidate lines, best first.

    This powers the live analysis board — it is re-run every time the
    position on *our own* board changes, and the UI draws an arrow for
    line 1 (plus fainter arrows for lines 2-3).
    """
    results: list[EngineResult] = engine.analyse_multipv(
        board, depth=depth, multipv=multipv
    )
    lines = [_line_to_dict(board, r, i + 1) for i, r in enumerate(results)]
    best = results[0] if results else None
    eval_white = (
        (best.score_cp if board.turn == chess.WHITE else -best.score_cp)
        if best
        else 0
    )
    return {
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "eval_white_cp": eval_white,
        "mate_white": (
            (best.mate if board.turn == chess.WHITE else -best.mate)
            if best and best.mate is not None
            else None
        ),
        "win_percent_white": round(win_percent(eval_white), 1),
        "best_move_uci": best.best_move_uci if best else None,
        "best_move_san": (
            board.san(best.best_move) if best and best.best_move else None
        ),
        "lines": lines,
        "depth": best.depth if best else depth,
        "engine": engine.name,
        "is_check": board.is_check(),
        "is_game_over": board.is_game_over(),
    }


def evaluate_guess(engine, board: chess.Board, move: chess.Move, *, depth: int = 14) -> dict:
    """Score a trainee's guessed move against the engine's top lines.

    'Guess the best move' training: the user commits to a move *before*
    seeing any engine output; we then report whether it matched the
    engine's first choice, was in the top 3, or how much Win% it lost —
    plus the lines themselves, revealed after the guess.
    """
    if move not in board.legal_moves:
        raise ValueError("این حرکت در این وضعیت قانونی نیست.")
    multi = analyze_position_multipv(engine, board, depth=depth, multipv=3)
    guessed_san = board.san(move)
    rank = None
    for line in multi["lines"]:
        if line["move_uci"] == move.uci():
            rank = line["rank"]
            eval_after_cp = line["eval_cp"]
            break
    else:
        # Not in the top 3: evaluate the position after the guessed move.
        board.push(move)
        after: EngineResult = engine.analyse(board, depth=depth)
        board.pop()
        eval_after_cp = -after.score_cp

    best_eval = multi["lines"][0]["eval_cp"] if multi["lines"] else eval_after_cp
    is_best = rank == 1
    win_before, win_after = win_percent(best_eval), win_percent(eval_after_cp)
    win_loss = 0.0 if is_best else max(0.0, win_before - win_after)
    classification = classify(is_best, win_loss)
    return {
        "fen": board.fen(),
        "guessed_san": guessed_san,
        "guessed_uci": move.uci(),
        "rank": rank,  # 1..3 or None if outside the engine's top 3
        "is_engine_best": is_best,
        "classification": classification,
        "classification_fa": LABELS_FA[classification],
        "accuracy": 100.0 if is_best else round(move_accuracy(win_before, win_after), 1),
        "cp_loss": 0 if is_best else int(max(0, best_eval - eval_after_cp)),
        "win_before": round(win_before, 1),
        "win_after": round(max(win_after, win_before) if is_best else win_after, 1),
        "win_loss": round(win_loss, 1),
        "best_move_san": multi["best_move_san"],
        "best_move_uci": multi["best_move_uci"],
        "lines": multi["lines"],  # revealed only after the guess — the lesson
        "engine": multi["engine"],
        "depth": multi["depth"],
    }


def analyze_game(
    engine,
    game: "chess.pgn.Game",  # noqa: F821 - chess.pgn is part of python-chess
    *,
    depth: int = 14,
) -> GameAnalysis:
    """Analyse every move of a finished game.

    Performance note: the evaluation *after* ply N is the evaluation
    *before* ply N+1 (same board, opposite side to move), so we carry it
    over instead of paying for a second engine call per move.
    """
    import chess.pgn  # local import keeps the type reference honest

    board = game.board()
    moves: list[MoveAnalysis] = []
    # cached result for the current position (side-to-move POV)
    current: EngineResult = engine.analyse(board, depth=depth)

    for ply, move in enumerate(game.mainline_moves()):
        fen_before = board.fen()
        san = board.san(move)
        mover_is_white = board.turn == chess.WHITE
        eval_before = current.score_cp  # mover's POV
        best_san = board.san(current.best_move) if current.best_move else None
        best_line = _pv_san(board, current.pv)

        board.push(move)
        nxt: EngineResult = engine.analyse(board, depth=depth)
        # nxt is scored for the *opponent*; flip it back to the mover.
        eval_after_mover = -nxt.score_cp
        eval_after_white = nxt.score_cp if board.turn == chess.WHITE else -nxt.score_cp

        cp_loss = max(0, eval_before - eval_after_mover)
        win_before, win_after = win_percent(eval_before), win_percent(eval_after_mover)
        win_loss = max(0.0, win_before - win_after)
        is_best = current.best_move is not None and move == current.best_move
        if is_best:
            # You cannot "lose" anything by playing the engine's own move.
            # (Eval-after can still differ from eval-before because the
            # second search looks one ply deeper — a horizon artefact
            # that must not be charged to the player.)
            cp_loss, win_loss, win_after = 0, 0.0, max(win_after, win_before)

        moves.append(
            MoveAnalysis(
                ply=ply,
                san=san,
                uci=move.uci(),
                color="white" if mover_is_white else "black",
                fen_before=fen_before,
                eval_before_cp=eval_before,
                eval_after_cp=eval_after_mover,
                eval_after_white_cp=eval_after_white,
                best_move_san=best_san,
                best_move_uci=current.best_move_uci,
                best_line_san=best_line,
                cp_loss=int(cp_loss),
                win_before=win_before,
                win_after=win_after,
                win_loss=win_loss,
                accuracy=move_accuracy(win_before, win_after),
                classification=classify(is_best, win_loss),
            )
        )
        current = nxt  # reuse for the next ply

    headers = {k: v for k, v in game.headers.items()}
    return GameAnalysis(
        moves=moves,
        engine_name=engine.name,
        depth=depth,
        headers=headers,
        result=headers.get("Result", "*"),
    )
