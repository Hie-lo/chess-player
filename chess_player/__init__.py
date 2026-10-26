"""chess-player: post-game / offline-training chess analysis.

This package analyses *finished* games (PGN), manually entered positions
(FEN) and offline games played against the built-in engine. It deliberately
does NOT capture the screen of a live online game or suggest moves during
one — engine assistance during rated online play is cheating under the
Chess.com Fair Play Policy and the Lichess Terms of Service.
"""

from .analysis import (
    GameAnalysis,
    MoveAnalysis,
    analyze_game,
    analyze_position,
    analyze_position_multipv,
    evaluate_guess,
)
from .engine import (
    EngineResult,
    SimpleEngine,
    StockfishEngine,
    find_stockfish,
    get_engine,
    stockfish_candidates,
)

__all__ = [
    "EngineResult",
    "GameAnalysis",
    "MoveAnalysis",
    "SimpleEngine",
    "StockfishEngine",
    "analyze_game",
    "analyze_position",
    "analyze_position_multipv",
    "evaluate_guess",
    "find_stockfish",
    "get_engine",
    "stockfish_candidates",
]

__version__ = "0.1.0"
