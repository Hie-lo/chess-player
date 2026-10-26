import json
import threading
import urllib.request

from chess_player.engine import SimpleEngine
from chess_player.webapp import make_server


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
        assert "تمرین آفلاین" in page

        health = _request(port, "GET", "/api/health")
        assert health["ok"]

        game = _request(port, "POST", "/api/new_game", {})
        assert "e2e4" in game["legal"]

        after = _request(port, "POST", "/api/move", {"fen": game["fen"], "uci": "e2e4"})
        assert after["feedback"]["your_move_san"] == "e4"
        assert after["engine_move_san"]  # engine replied
        assert after["feedback"]["classification"] in (
            "best", "excellent", "good", "inaccuracy", "mistake", "blunder",
        )

        fen_info = _request(
            port, "POST", "/api/analyze_fen",
            {"fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"},
        )
        assert fen_info["best_move_uci"]
    finally:
        server.shutdown()
        server.server_close()
