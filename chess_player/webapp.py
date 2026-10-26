"""Live analysis web app (stdlib only — no Flask needed).

One page, one board, three training modes (the board itself lives in
this app — it is not connected to any external chess site):

1. **Live analysis board** — move pieces for either side and the engine
   (Stockfish when installed) re-evaluates the position immediately:
   eval bar, best-move arrow and the top-3 lines (MultiPV).
2. **Play vs. the engine** — you commit a move first, then get feedback
   on it and the engine replies.
3. **Guess the best move** — engine output stays hidden until you guess;
   then your move is scored against the engine's top lines.

Plus: post-game PGN analysis / step-through, FEN loading, and position
import from an image the user manually supplies (they mark the board
corners, verify the recognised pieces on an editor, then analysis runs).

The server is stateless: the browser keeps the current FEN + history and
sends the FEN with every request; python-chess re-validates server-side.

Run:  python -m chess_player serve --port 8000   ->  http://localhost:8000
"""

from __future__ import annotations

import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import chess
import chess.pgn

from .analysis import (
    LABELS_FA,
    analyze_game,
    analyze_position,
    analyze_position_multipv,
    classify,
    evaluate_guess,
    move_accuracy,
    win_percent,
)
from .engine import get_engine

# One engine process (Stockfish) cannot serve threads concurrently.
_ENGINE_LOCK = threading.Lock()
_PAGE_PATH = Path(__file__).parent / "static" / "index.html"
_MAX_IMAGE_BYTES = 8 * 1024 * 1024


def _legal_uci(board: chess.Board) -> list[str]:
    return [m.uci() for m in board.legal_moves]


def _game_status(board: chess.Board) -> str | None:
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "draw_insufficient"
    if board.can_claim_draw():
        return "draw_claimable"
    return None


def _board_state(board: chess.Board) -> dict:
    return {
        "fen": board.fen(),
        "legal": _legal_uci(board),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "status": _game_status(board),
        "in_check": board.is_check(),
        "is_game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
    }


def _parse_move(board: chess.Board, uci: str) -> chess.Move:
    try:
        move = chess.Move.from_uci(uci)
    except ValueError as exc:
        raise ValueError(f"حرکت نامعتبر: {exc}") from exc
    if move not in board.legal_moves:
        raise ValueError("این حرکت در این وضعیت قانونی نیست.")
    return move


def _parse_board(fen: str) -> chess.Board:
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"FEN نامعتبر: {exc}") from exc
    if not board.is_valid():
        raise ValueError(
            "وضعیت قانونی نیست (مثلاً شاه‌ها یا کیشِ نام ممکن) — "
            f"کد وضعیت: {board.status()}"
        )
    return board


class _App:
    """All request logic, independent of HTTP (keeps it unit-testable)."""

    def __init__(self, engine) -> None:
        self.engine = engine
        self.is_fallback = "SimpleEngine" in engine.name
        # Fallback engine: keep interactive depth low so replies are snappy.
        self.depth = 2 if self.is_fallback else 12

    def _depth(self, requested: object) -> int:
        try:
            depth = int(requested)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return self.depth
        return max(1, min(depth, 3 if self.is_fallback else 24))

    # -- basic board plumbing -------------------------------------------
    def new_game(self) -> dict:
        return _board_state(chess.Board())

    def board_state(self, fen: str) -> dict:
        return _board_state(_parse_board(fen))

    def apply_move(self, fen: str, uci: str) -> dict:
        """Apply a move for whichever side is to move (analysis board)."""
        board = _parse_board(fen)
        move = _parse_move(board, uci)
        san = board.san(move)
        board.push(move)
        state = _board_state(board)
        state["san"] = san
        state["uci"] = uci
        return state

    # -- live analysis ----------------------------------------------------
    def live_analyze(self, fen: str, depth: object = None, multipv: int = 3) -> dict:
        board = _parse_board(fen)
        with _ENGINE_LOCK:
            result = analyze_position_multipv(
                self.engine, board, depth=self._depth(depth), multipv=multipv
            )
        result["legal"] = _legal_uci(board)
        result["status"] = _game_status(board)
        return result

    def guess(self, fen: str, uci: str, depth: object = None) -> dict:
        board = _parse_board(fen)
        move = _parse_move(board, uci)
        with _ENGINE_LOCK:
            return evaluate_guess(self.engine, board, move, depth=self._depth(depth))

    # -- play vs engine ---------------------------------------------------
    def human_move(self, fen: str, uci: str) -> dict:
        board = _parse_board(fen)
        move = _parse_move(board, uci)
        if board.turn != chess.WHITE:
            raise ValueError("در حالت بازی شما با سفید بازی می‌کنید؛ نوبت سفید نیست.")

        with _ENGINE_LOCK:
            before = self.engine.analyse(board, depth=self.depth)
            san = board.san(move)
            best_san = board.san(before.best_move) if before.best_move else None
            board.push(move)

            after = self.engine.analyse(board, depth=self.depth)  # engine to move
            eval_after_human = -after.score_cp
            cp_loss = max(0, before.score_cp - eval_after_human)
            win_before = win_percent(before.score_cp)
            win_after = win_percent(eval_after_human)
            win_loss = max(0.0, win_before - win_after)
            is_best = before.best_move == move
            classification = classify(is_best, win_loss)
            feedback = {
                "your_move_san": san,
                "classification": classification,
                "classification_fa": LABELS_FA[classification],
                "accuracy": (
                    100.0 if is_best else round(move_accuracy(win_before, win_after), 1)
                ),
                "cp_loss": 0 if is_best else int(cp_loss),
                "win_before": round(win_before, 1),
                "win_after": round(win_after, 1),
                # Shown only AFTER the move is committed — training feedback.
                "engine_preferred_san": best_san,
            }

            engine_move_san = None
            status = _game_status(board)
            if status is None and after.best_move is not None:
                engine_move_san = board.san(after.best_move)
                board.push(after.best_move)
                status = _game_status(board)
                final = self.engine.analyse(board, depth=self.depth)
                eval_white, final_mate = final.score_cp, final.mate
            else:
                eval_white, final_mate = eval_after_human, after.mate

        state = _board_state(board)
        state.update(
            {
                "feedback": feedback,
                "engine_move_san": engine_move_san,
                "san": san,  # the human move that led here (engine replied after)
                "eval_white_cp": int(eval_white),
                "mate": final_mate,
                "status": status,
            }
        )
        return state

    # -- PGN ---------------------------------------------------------------
    def analyze_pgn_text(self, pgn_text: str) -> dict:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None or not any(game.mainline_moves()):
            raise ValueError("PGN خالی یا نامعتبر است.")
        with _ENGINE_LOCK:
            analysis = analyze_game(self.engine, game, depth=self.depth)
        return analysis.to_dict()

    def load_pgn_positions(self, pgn_text: str) -> dict:
        """Every position of a PGN, for step-through with live analysis."""
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            raise ValueError("PGN خالی یا نامعتبر است.")
        board = game.board()
        positions = [{"fen": board.fen(), "san": None, "legal": _legal_uci(board)}]
        for move in game.mainline_moves():
            san = board.san(move)
            board.push(move)
            positions.append(
                {"fen": board.fen(), "san": san, "legal": _legal_uci(board)}
            )
        if len(positions) == 1:
            raise ValueError("این PGN هیچ حرکتی ندارد.")
        return {
            "headers": dict(game.headers),
            "positions": positions,
            "result": game.headers.get("Result", "*"),
        }

    def analyze_fen(self, fen: str) -> dict:
        board = _parse_board(fen)
        with _ENGINE_LOCK:
            return analyze_position(self.engine, board, depth=self.depth)

    # -- image import (manually supplied, verified by the user) -----------
    def fen_from_image(self, image_data_url: str) -> dict:
        from .image_fen import ImageImportUnavailable, board_grid_from_image, fen_from_grid

        payload = image_data_url.split(",", 1)[-1] if "," in image_data_url[:64] else image_data_url
        try:
            raw = base64.b64decode(payload, validate=True)
        except Exception as exc:
            raise ValueError(f"دادهٔ تصویر نامعتبر است: {exc}") from exc
        if len(raw) > _MAX_IMAGE_BYTES:
            raise ValueError("تصویر بزرگ‌تر از حد مجاز (۸ مگابایت) است.")
        try:
            proposal = board_grid_from_image(raw)
        except ImageImportUnavailable:
            raise
        except Exception as exc:
            raise ValueError(f"خواندن تصویر ناموفق بود: {exc}") from exc
        proposal["fen_guess"] = fen_from_grid(proposal["grid"], "white")
        return proposal

    def fen_from_grid(self, grid: list, turn: str) -> dict:
        from .image_fen import fen_from_grid

        if (
            not isinstance(grid, list)
            or len(grid) != 8
            or any(not isinstance(r, list) or len(r) != 8 for r in grid)
            or any(c not in ("", *"PNBRQKpnbrqk") for row in grid for c in row)
        ):
            raise ValueError("گرید تخته باید ۸×۸ و شامل حروف مهره‌های FEN باشد.")
        turn = "black" if str(turn).lower().startswith("b") else "white"
        fen = fen_from_grid(grid, turn)
        board = chess.Board(fen)
        valid = board.is_valid()
        return {
            "fen": fen,
            "valid": valid,
            "status": int(board.status()),
            "legal": _legal_uci(board) if valid else [],
            "message": (
                "وضعیت معتبر است."
                if valid
                else "وضعیت قانونی نیست — شاه‌ها، پیاده‌های ردیف اول/هشتم و کیش نام ممکن را در ویرایشگر اصلاح کنید."
            ),
        }


def _load_page() -> bytes:
    try:
        return _PAGE_PATH.read_bytes()
    except OSError:  # pragma: no cover
        return "<!doctype html><meta charset=utf-8><h1>index.html missing</h1>".encode()


def make_handler(app: _App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            if length > _MAX_IMAGE_BYTES * 2:  # base64 inflates payloads
                raise ValueError("درخواست بزرگ‌تر از حد مجاز است.")
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self):  # noqa: N802 (http.server naming)
            if self.path == "/":
                body = _load_page()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/health":
                self._send_json(
                    {
                        "ok": True,
                        "engine": app.engine.name,
                        "is_fallback_engine": app.is_fallback,
                        "suggested_depth": app.depth,
                    }
                )
            else:
                self._send_json({"error": "not found"}, 404)

        def do_HEAD(self):  # noqa: N802 — proxies/health-checkers use HEAD
            if self.path in ("/", "/api/health"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

        def do_POST(self):  # noqa: N802
            try:
                data = self._read_json()
                routes = {
                    "/api/new_game": lambda: app.new_game(),
                    "/api/board_state": lambda: app.board_state(str(data["fen"])),
                    "/api/apply_move": lambda: app.apply_move(
                        str(data["fen"]), str(data["uci"])
                    ),
                    "/api/live_analyze": lambda: app.live_analyze(
                        str(data["fen"]),
                        data.get("depth"),
                        int(data.get("multipv", 3)),
                    ),
                    "/api/guess": lambda: app.guess(
                        str(data["fen"]), str(data["uci"]), data.get("depth")
                    ),
                    "/api/move": lambda: app.human_move(
                        str(data["fen"]), str(data["uci"])
                    ),
                    "/api/analyze_pgn": lambda: app.analyze_pgn_text(str(data["pgn"])),
                    "/api/load_pgn": lambda: app.load_pgn_positions(str(data["pgn"])),
                    "/api/analyze_fen": lambda: app.analyze_fen(str(data["fen"])),
                    "/api/fen_from_image": lambda: app.fen_from_image(
                        str(data["image"])
                    ),
                    "/api/fen_from_grid": lambda: app.fen_from_grid(
                        data["grid"], str(data.get("turn", "white"))
                    ),
                }
                handler = routes.get(self.path)
                if handler is None:
                    self._send_json({"error": "not found"}, 404)
                else:
                    self._send_json(handler())
            except (ValueError, KeyError) as exc:
                self._send_json({"error": str(exc)}, 400)
            except Exception as exc:
                # ImageImportUnavailable et al.: surface a clean message.
                status = 501 if type(exc).__name__ == "ImageImportUnavailable" else 500
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status)

    return Handler


def make_server(host: str, port: int, engine) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(_App(engine)))


def serve(host: str = "0.0.0.0", port: int = 8000, engine_arg: str = "auto") -> None:
    engine = get_engine(engine_arg)
    server = make_server(host, port, engine)  # raises OSError if port is busy
    # 0.0.0.0 is a *bind* address ("listen on all interfaces"), not an
    # address a browser can open — print the URL that actually works.
    display_host = "localhost" if host in ("0.0.0.0", "::", "") else host
    print(f"موتور: {engine.name}", flush=True)
    if "SimpleEngine" in engine.name and engine_arg != "simple":
        from .engine import stockfish_candidates

        candidates = stockfish_candidates()
        print(
            "ℹ️ Stockfish پیدا نشد/اجرا نشد و موتور ضعیف جایگزین فعال است.\n"
            "   ساده‌ترین راه — نصب خودکار موتور واقعی:\n"
            "     python -m chess_player install-stockfish\n"
            "   (توجه: ‎pip install stockfish‎ فقط wrapper پایتون است و موتور ندارد.)\n"
            "   مسیرهای بررسی‌شده با دستور `python -m chess_player doctor` دیده می‌شوند."
            + (f"\n   نامزدهای پیدا‌شده: {candidates}" if candidates else ""),
            flush=True,
        )
    print(
        f"استودیو تحلیل زنده: http://{display_host}:{port}  "
        f"(همین آدرس را در مرورگر باز کنید — نه 0.0.0.0)  (Ctrl+C برای خروج)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.close()
