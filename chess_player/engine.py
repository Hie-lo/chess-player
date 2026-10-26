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

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

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
        return self._info_to_result(board, info, depth)

    def _info_to_result(self, board: chess.Board, info, depth: int) -> EngineResult:
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

    def analyse_multipv(
        self,
        board: chess.Board,
        *,
        depth: int = 14,
        multipv: int = 3,
        time_limit: float | None = None,
    ) -> list[EngineResult]:
        """Top-N lines (UCI ``setoption name MultiPV``), best first."""
        terminal = _terminal_result(board, self.name)
        if terminal is not None:
            return [terminal]
        limit = (
            chess.engine.Limit(time=time_limit)
            if time_limit is not None
            else chess.engine.Limit(depth=depth)
        )
        infos = self._engine.analyse(board, limit, multipv=max(1, multipv))
        return [self._info_to_result(board, info, depth) for info in infos]

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
        return self._to_result(score, pv, depth)

    def _to_result(self, score: int, pv: list[chess.Move], depth: int) -> EngineResult:
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
            pv=list(pv),
            depth=depth,
            engine_name=self.name,
        )

    def analyse_multipv(
        self,
        board: chess.Board,
        *,
        depth: int = 2,
        multipv: int = 3,
        time_limit: float | None = None,  # API compatibility; ignored
    ) -> list[EngineResult]:
        """Top-N lines by searching every root move (best first).

        This costs roughly one search per legal move, so the web app uses
        depth 2 for the fallback engine; Stockfish does MultiPV natively
        inside a single search and is far cheaper per line.
        """
        terminal = _terminal_result(board, self.name)
        if terminal is not None:
            return [terminal]
        depth = max(1, min(depth, self.max_depth))
        scored: list[tuple[int, list[chess.Move]]] = []
        for move in _ordered_moves(board):
            board.push(move)
            if board.is_game_over():
                child_score, child_pv = _evaluate(board), []
            else:
                child_score, child_pv = self._negamax(
                    board, depth - 1, -MATE_SCORE - 1, MATE_SCORE + 1, 1
                )
            board.pop()
            scored.append((-child_score, [move, *child_pv]))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            self._to_result(score, pv, depth)
            for score, pv in scored[: max(1, multipv)]
        ]

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
# Factory + Stockfish discovery
# --------------------------------------------------------------------------

_ARCHIVE_SUFFIXES = {".zip", ".tar", ".gz", ".tgz", ".xz", ".7z", ".rar"}
_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache"}
_SYSTEM_PATHS = [
    "/usr/bin/stockfish",
    "/usr/local/bin/stockfish",
    "/opt/homebrew/bin/stockfish",
    "/usr/games/stockfish",
    r"C:\Program Files\Stockfish\stockfish.exe",
    r"C:\Program Files (x86)\Stockfish\stockfish.exe",
]


def _variant_rank(name: str) -> int:
    """Prefer plain / widely-compatible builds over AVX-512 ones.

    An AVX512/VNNI512 binary crashes with an illegal instruction on CPUs
    without those extensions, so when a download folder offers several
    variants we try the safe ones first.
    """
    lowered = name.lower()
    if lowered in ("stockfish", "stockfish.exe"):
        return 0
    if "avx512" in lowered or "vnni512" in lowered or "icl" in lowered:
        return 5
    if "avxvnni" in lowered or "vnni" in lowered:
        return 4
    if "bmi2" in lowered:
        return 3
    if "avx2" in lowered:
        return 2
    if "modern" in lowered or "popcnt" in lowered or "sse41" in lowered:
        return 1
    return 2


def _is_plausible_binary(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() in _ARCHIVE_SUFFIXES:
        return False  # a downloaded .zip must be extracted first
    # Accept bare ``stockfish`` (Unix) and ``stockfish....exe`` (Windows) —
    # but not stockfish.py / .md / .txt / source files that share the stem.
    return path.suffix.lower() in ("", ".exe")


def stockfish_candidates(
    explicit_path: str | None = None, *, roots: list[Path] | None = None
) -> list[str]:
    """Every plausible Stockfish binary, most promising first.

    Search order: explicit path -> ``$STOCKFISH_PATH`` -> ``PATH`` ->
    well-known system locations -> the project folder, recursively
    (``./stockfish.exe``, ``./engines/...``, ``./stockfish/stockfish-*.exe``
    and friends) -> the user's ``Downloads`` folder (the browser's
    default save location — a common place for the extracted engine to
    be hiding) -> any ``stockfish*.exe`` bundled inside site-packages.
    Dropping the extracted binary anywhere in the project is therefore
    enough — run ``python -m chess_player doctor`` to see exactly what
    was found, or ``install-stockfish`` to fetch the real engine.
    """
    found: list[str] = []

    def add(value: str | Path | None) -> None:
        if not value:
            return
        text = str(value)
        if text not in found:
            found.append(text)

    add(explicit_path)
    add(os.environ.get("STOCKFISH_PATH"))
    add(shutil.which("stockfish"))
    add(shutil.which("stockfish.exe"))
    for system_path in _SYSTEM_PATHS:
        if Path(system_path).exists():
            add(system_path)

    search_roots: list[Path] = list(roots) if roots is not None else []
    if roots is None:
        project_root = Path(__file__).resolve().parent.parent
        search_roots = [Path.cwd(), project_root, Path.home() / "Downloads"]
        try:
            import sysconfig

            purelib = sysconfig.get_path("purelib")
            if purelib:
                search_roots.append(Path(purelib))
        except Exception:
            pass

    globbed: list[Path] = []
    for root in search_roots:
        if not root.is_dir():
            continue
        for pattern in ("stockfish*", "**/stockfish*"):
            try:
                matches = root.glob(pattern)
            except OSError:
                continue
            for match in matches:
                # Skip noise dirs *relative to this root* (so searching
                # a venv's site-packages explicitly still works).
                try:
                    rel_parts = match.relative_to(root).parts
                except ValueError:
                    rel_parts = match.parts
                if any(part in _SKIP_DIRS for part in rel_parts):
                    continue
                if match not in globbed and _is_plausible_binary(match):
                    globbed.append(match)
    # Shallower, plainer, safer-CPU builds first.
    globbed.sort(
        key=lambda p: (len(p.parts), _variant_rank(p.name), p.name.lower())
    )
    for match in globbed:
        add(match)
    return found


def find_stockfish(explicit_path: str | None = None) -> str | None:
    """First plausible Stockfish binary (see :func:`stockfish_candidates`)."""
    candidates = stockfish_candidates(explicit_path)
    return candidates[0] if candidates else None


def _try_stockfish(path: str) -> StockfishEngine:
    """Launch a candidate, fixing a missing +x bit once along the way."""
    try:
        return StockfishEngine(path)
    except PermissionError:
        candidate = Path(path)
        if candidate.is_file() and os.name != "nt":
            candidate.chmod(candidate.stat().st_mode | 0o755)
            return StockfishEngine(path)
        raise


def get_engine(engine_arg: str | None = "auto"):
    """Create an engine.

    ``engine_arg``:
      * ``"auto"`` (default) — first working Stockfish found by
        :func:`stockfish_candidates`, else SimpleEngine (candidates that
        fail to launch are skipped with a warning instead of crashing);
      * ``"simple"`` — always the fallback engine;
      * any other string — a path to a UCI engine binary (must work).
    """
    if engine_arg == "simple":
        return SimpleEngine()
    if engine_arg in (None, "auto"):
        failures: list[str] = []
        for candidate in stockfish_candidates():
            # A bare name resolved via PATH exists by construction;
            # path-like candidates must exist as files.
            if os.sep in candidate or "/" in candidate or "\\" in candidate:
                if not Path(candidate).is_file():
                    continue
            try:
                return _try_stockfish(candidate)
            except Exception as exc:  # wrong arch, missing DLL, not UCI...
                failures.append(f"{candidate} -> {type(exc).__name__}: {exc}")
        for failure in failures:
            print(f"⚠️ Stockfish پیدا شد ولی اجرا نشد: {failure}", file=sys.stderr)
        if failures:
            print(
                "⚠️ از موتور جایگزین استفاده می‌شود. "
                "`python -m chess_player doctor` را برای جزئیات اجرا کنید.",
                file=sys.stderr,
            )
        return SimpleEngine()
    return _try_stockfish(engine_arg)
