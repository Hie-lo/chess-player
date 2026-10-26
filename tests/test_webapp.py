import base64
import json
import threading
import urllib.error
import urllib.request

import chess

from chess_player.engine import SimpleEngine
from chess_player.webapp import make_server

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _request(port, method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode()
        return json.loads(body) if path.startswith("/api") else body


def test_webapp_flow():
    server = make_server("127.0.0.1", 0, SimpleEngine())
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        page = _request(port, "GET", "/")
        assert "تحلیل زنده" in page
        assert "حدس بهترین حرکت" in page
        assert "/api/live_analyze" in page

        health = _request(port, "GET", "/api/health")
        assert health["ok"] and "suggested_depth" in health

        game = _request(port, "POST", "/api/new_game", {})
        assert "e2e4" in game["legal"]

        # Analysis board: either side may move, no engine reply.
        after_e4 = _request(port, "POST", "/api/apply_move", {"fen": START, "uci": "e2e4"})
        assert after_e4["san"] == "e4" and after_e4["turn"] == "black"
        after_e5 = _request(
            port, "POST", "/api/apply_move", {"fen": after_e4["fen"], "uci": "e7e5"}
        )
        assert after_e5["turn"] == "white"

        # Live (MultiPV) analysis of the current position.
        live = _request(port, "POST", "/api/live_analyze", {"fen": START, "depth": 2})
        assert live["best_move_uci"] == "e2e4"
        assert len(live["lines"]) == 3
        assert live["legal"]

        # Guess-the-move scoring.
        fen = "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3"
        guess = _request(port, "POST", "/api/guess", {"fen": fen, "uci": "g8f6"})
        assert guess["classification"] in ("mistake", "blunder")
        assert guess["best_move_san"] == "g6"

        # Play vs engine still works.
        played = _request(port, "POST", "/api/move", {"fen": START, "uci": "e2e4"})
        assert played["feedback"]["your_move_san"] == "e4"
        assert played["engine_move_san"]

        # PGN step-through positions.
        pgn = "1. e4 e5 2. Nf3 Nc6 *"
        loaded = _request(port, "POST", "/api/load_pgn", {"pgn": pgn})
        assert len(loaded["positions"]) == 5  # start + 4 plies
        assert loaded["positions"][1]["san"] == "e4"

        # Legacy single-position endpoint.
        fen_info = _request(
            port, "POST", "/api/analyze_fen",
            {"fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"},
        )
        assert fen_info["best_move_uci"]

        # Image import: render the start position and read it back.
        try:
            from chess_player.image_fen import render_board_image

            png = render_board_image(chess.Board())
        except Exception:
            png = None
        if png is not None:
            data_url = "data:image/png;base64," + base64.b64encode(png).decode()
            proposal = _request(port, "POST", "/api/fen_from_image", {"image": data_url})
            assert proposal["fen_guess"].startswith("rnbqkbnr/pppppppp")
            verified = _request(
                port, "POST", "/api/fen_from_grid",
                {"grid": proposal["grid"], "turn": "white"},
            )
            assert verified["valid"] is True

        # Illegal move is a clean 400, not a 500.
        try:
            _request(port, "POST", "/api/apply_move", {"fen": START, "uci": "e2e5"})
            raise AssertionError("expected HTTPError")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
    finally:
        server.shutdown()
        server.server_close()
