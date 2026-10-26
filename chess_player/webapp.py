"""Offline training web app (stdlib only — no Flask needed).

Two legal training tools in one page:

1. **Play vs. the engine offline.** You commit your move first; only then
   do you get feedback (eval, classification, what the engine preferred).
   That "move first, feedback after" order is what makes it training
   rather than assistance.
2. **Post-game PGN analysis.** Paste the PGN of a finished game and get
   the same accuracy / blunder report as the CLI.

The server is stateless: the browser keeps the current FEN and sends it
with every request, and python-chess re-validates everything server-side.

Run:  python -m chess_player serve --port 8000   ->  http://localhost:8000
"""

from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import chess
import chess.pgn

from .analysis import (
    LABELS_FA,
    analyze_game,
    analyze_position,
    classify,
    move_accuracy,
    win_percent,
)
from .engine import get_engine

# One engine process (Stockfish) cannot serve threads concurrently.
_ENGINE_LOCK = threading.Lock()


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


class _App:
    """All request logic, independent of HTTP (keeps it unit-testable)."""

    def __init__(self, engine) -> None:
        self.engine = engine
        # Fallback engine: keep interactive depth low so replies are snappy.
        self.depth = 2 if "SimpleEngine" in engine.name else 12

    def new_game(self) -> dict:
        board = chess.Board()
        return {"fen": board.fen(), "legal": _legal_uci(board), "turn": "white"}

    def human_move(self, fen: str, uci: str) -> dict:
        board = chess.Board(fen)
        try:
            move = chess.Move.from_uci(uci)
        except ValueError as exc:
            raise ValueError(f"حرکت نامعتبر: {exc}") from exc
        if move not in board.legal_moves:
            raise ValueError("این حرکت در این وضعیت قانونی نیست.")
        if board.turn != chess.WHITE:
            raise ValueError("در حالت تمرینی شما با سفید بازی می‌کنید؛ نوبت سفید نیست.")

        with _ENGINE_LOCK:
            before = self.engine.analyse(board, depth=self.depth)
            san = board.san(move)
            best_san = board.san(before.best_move) if before.best_move else None
            board.push(move)

            after = self.engine.analyse(board, depth=self.depth)  # engine to move
            eval_after_human = -after.score_cp
            cp_loss = max(0, before.score_cp - eval_after_human)
            win_before, win_after = win_percent(before.score_cp), win_percent(eval_after_human)
            win_loss = max(0.0, win_before - win_after)
            is_best = before.best_move == move
            classification = classify(is_best, win_loss)
            feedback = {
                "your_move_san": san,
                "classification": classification,
                "classification_fa": LABELS_FA[classification],
                "accuracy": 100.0 if is_best else round(move_accuracy(win_before, win_after), 1),
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
                # board.turn is White again here -> score is already White POV
                eval_white = final.score_cp
                final_mate = final.mate
            else:
                eval_white = eval_after_human
                final_mate = after.mate

        return {
            "fen": board.fen(),
            "legal": _legal_uci(board),
            "turn": "white" if board.turn == chess.WHITE else "black",
            "feedback": feedback,
            "engine_move_san": engine_move_san,
            "eval_white_cp": int(eval_white),
            "mate": final_mate,
            "status": status,
            "in_check": board.is_check(),
        }

    def analyze_pgn_text(self, pgn_text: str) -> dict:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None or not any(game.mainline_moves()):
            raise ValueError("PGN خالی یا نامعتبر است.")
        with _ENGINE_LOCK:
            analysis = analyze_game(self.engine, game, depth=self.depth)
        return analysis.to_dict()

    def analyze_fen(self, fen: str) -> dict:
        board = chess.Board(fen)
        if not board.is_valid():
            raise ValueError(f"وضعیت غیرقانونی: {board.status()}")
        with _ENGINE_LOCK:
            return analyze_position(self.engine, board, depth=self.depth)


PAGE = r"""<!doctype html>
<html lang="fa" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>مربی شطرنج — تمرین آفلاین و تحلیل پس از بازی</title>
<style>
body{font-family:Tahoma,sans-serif;background:#161512;color:#eee;max-width:1000px;margin:20px auto;padding:0 16px}
h1{font-size:22px}.card{background:#24231f;border-radius:12px;padding:16px;margin:14px 0}
.warn{background:#3a2c12;border:1px solid #a67c00;border-radius:10px;padding:10px;font-size:14px;line-height:1.9}
#board{display:grid;grid-template-columns:repeat(8,1fr);width:min(480px,90vw);aspect-ratio:1;border:3px solid #444;border-radius:6px;overflow:hidden;direction:ltr}
.sq{display:flex;align-items:center;justify-content:center;font-size:clamp(26px,6vw,44px);cursor:pointer;user-select:none}
.light{background:#ecdcb9;color:#000}.dark{background:#b58863;color:#000}
.sel{outline:3px solid #4caf50;outline-offset:-3px}.hint{box-shadow:inset 0 0 0 4px rgba(76,175,80,.55)}
#evalbar{width:min(480px,90vw);height:18px;background:#333;border-radius:9px;overflow:hidden;margin:8px 0}
#evalfill{height:100%;background:#eee;width:50%;transition:width .4s}
button{background:#4caf50;color:#fff;border:0;border-radius:8px;padding:9px 18px;font-size:15px;cursor:pointer;font-family:inherit}
button:hover{background:#43a047}textarea{width:100%;min-height:110px;background:#111;color:#eee;border:1px solid #555;border-radius:8px;padding:8px;direction:ltr;text-align:left}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #3a3a3a;padding:4px;text-align:center;font-size:13px}
.badge{color:#fff;padding:2px 8px;border-radius:10px;font-size:12px}
.mono{direction:ltr;unicode-bidi:embed;font-family:monospace}
</style></head><body>
<h1>♞ مربی شطرنج — تمرین آفلاین + تحلیل پس از بازی</h1>
<div class="warn">⚠️ <b>بازی جوانمردانه:</b> این ابزار فقط برای تمرین آفلاین و بررسی
بازی‌های <b>تمام‌شده</b> است. استفاده از موتور هنگام بازی آنلاین (Chess.com / Lichess)
تقلب محسوب می‌شود و به بسته‌شدن حساب می‌انجامد — این برنامه هیچ اتصالی به آن سایت‌ها ندارد.</div>

<div class="card">
<h2>۱) بازی تمرینی مقابل موتور (شما سفید هستید)</h2>
<p>اول حرکت کنید؛ <b>بعد</b> بازخورد می‌گیرید: کیفیت حرکت، دقت، و این‌که موتور چه حرکتی را ترجیح می‌داد. روی مهره و بعد خانهٔ مقصد کلیک کنید. ارتقای پیاده خودکار به وزیر انجام می‌شود.</p>
<div id="evalbar"><div id="evalfill"></div></div>
<div id="evaltext">ارزیابی: —</div><br>
<div id="board"></div>
<p id="feedback">بازی جدید را شروع کنید.</p>
<button onclick="newGame()">بازی جدید</button>
<span id="engine-name" style="color:#aaa;font-size:13px"></span>
</div>

<div class="card">
<h2>۲) تحلیل بازی تمام‌شده (PGN)</h2>
<p>PGN بازی را از Chess.com یا Lichess دانلود/کپی کنید (Share → Export PGN) و اینجا بچسبانید:</p>
<textarea id="pgn" placeholder="[Event &quot;...&quot;]&#10;1. e4 e5 2. ..."></textarea><br><br>
<button onclick="analyzePgn()">تحلیل بازی</button>
<div id="pgn-result"></div>
</div>

<div class="card">
<h2>۳) تختهٔ تحلیل (FEN دستی)</h2>
<p>وضعیت یک پازل، کتاب، یا بازی تمام‌شده را با FEN وارد کنید:</p>
<textarea id="fen" style="min-height:50px" placeholder="rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"></textarea><br><br>
<button onclick="analyzeFen()">بهترین حرکت این وضعیت</button>
<div id="fen-result"></div>
</div>

<script>
const GLYPH={K:"\u2654",Q:"\u2655",R:"\u2656",B:"\u2657",N:"\u2658",P:"\u2659",k:"\u265A",q:"\u265B",r:"\u265C",b:"\u265D",n:"\u265E",p:"\u265F"};
let state={fen:"",legal:[]}, selected=null;
const $=id=>document.getElementById(id);
async function api(path,body){const r=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body||{})});const d=await r.json();if(!r.ok)throw new Error(d.error||r.status);return d;}
function squaresFromFen(fen){const rows=fen.split(" ")[0].split("/");const sq=[];for(const row of rows){for(const ch of row){if(/\d/.test(ch))for(let i=0;i<+ch;i++)sq.push("");else sq.push(ch);}}return sq;}
function sqName(idx){return "abcdefgh"[idx%8]+(8-Math.floor(idx/8));}
function render(){
  const sq=squaresFromFen(state.fen);const b=$("board");b.innerHTML="";
  sq.forEach((p,i)=>{const d=document.createElement("div");const f=i%8,r=Math.floor(i/8);
    d.className="sq "+(((f+r)%2===0)?"light":"dark");
    const name=sqName(i);
    if(selected===name)d.classList.add("sel");
    if(selected&&state.legal.some(u=>u.startsWith(selected+name)))d.classList.add("hint");
    d.textContent=p?GLYPH[p]:"";d.onclick=()=>onClick(name);b.appendChild(d);});
}
async function onClick(name){
  if(!selected){if(state.legal.some(u=>u.startsWith(name))){selected=name;render();}return;}
  const cand=state.legal.filter(u=>u.startsWith(selected+name));
  if(!cand.length){selected=state.legal.some(u=>u.startsWith(name))?name:null;render();return;}
  const uci=cand.find(u=>u.endsWith("q"))||cand[0];selected=null;
  $("feedback").textContent="موتور در حال فکر…";
  try{const d=await api("/api/move",{fen:state.fen,uci});
    state={fen:d.fen,legal:d.legal};render();
    const f=d.feedback,cp=d.eval_white_cp;
    $("evalfill").style.width=Math.max(2,Math.min(98,50+cp/10))+"%";
    $("evaltext").textContent="ارزیابی (دید سفید): "+(cp/100>=0?"+":"")+(cp/100).toFixed(2)+(d.mate?" (مات در "+Math.abs(d.mate)+")":"");
    let t="حرکت شما "+f.your_move_san+": "+f.classification_fa+" — دقت "+f.accuracy+"٪";
    if(f.engine_preferred_san&&f.classification!=="best")t+=" | موتور ترجیح می‌داد: "+f.engine_preferred_san;
    if(d.engine_move_san)t+=" | پاسخ موتور: "+d.engine_move_san;
    if(d.status)t+=" | وضعیت: "+d.status;
    $("feedback").textContent=t;
  }catch(e){$("feedback").textContent="خطا: "+e.message;render();}
}
async function newGame(){try{const d=await api("/api/new_game");state={fen:d.fen,legal:d.legal};selected=null;render();
  $("evalfill").style.width="50%";$("evaltext").textContent="ارزیابی: —";$("feedback").textContent="نوبت شما (سفید). موفق باشید!";}catch(e){alert(e.message);}}
const COLORS={best:"#2e7d32",excellent:"#43a047",good:"#7cb342",inaccuracy:"#f9a825",mistake:"#ef6c00",blunder:"#c62828"};
async function analyzePgn(){
  $("pgn-result").innerHTML="در حال تحلیل… (با موتور جایگزین ممکن است چند ثانیه طول بکشد)";
  try{const d=await api("/api/analyze_pgn",{pgn:$("pgn").value});
    let h="<p>دقت سفید: <b>"+d.white_accuracy+"٪</b> (ACPL "+d.white_acpl+") · دقت سیاه: <b>"+d.black_accuracy+"٪</b> (ACPL "+d.black_acpl+")<br><span style='color:#aaa'>"+d.engine+"</span></p>";
    h+="<table><tr><th>#</th><th>حرکت</th><th>ارزیابی</th><th>کیفیت</th><th>دقت</th><th>پیشنهاد موتور</th></tr>";
    for(const m of d.moves){const num=m.color==="white"?m.move_number+".":m.move_number+"…";
      h+="<tr><td>"+num+"</td><td><b>"+m.san+"</b></td><td class='mono'>"+(m.eval_after_white_cp/100).toFixed(2)+"</td><td><span class='badge' style='background:"+COLORS[m.classification]+"'>"+m.classification_fa+"</span></td><td>"+m.accuracy+"٪</td><td>"+(m.best_move_san||"—")+"</td></tr>";}
    $("pgn-result").innerHTML=h+"</table>";
  }catch(e){$("pgn-result").textContent="خطا: "+e.message;}
}
async function analyzeFen(){
  $("fen-result").textContent="در حال تحلیل…";
  try{const d=await api("/api/analyze_fen",{fen:$("fen").value.trim()});
    $("fen-result").innerHTML="<p>بهترین حرکت: <b>"+d.best_move_san+"</b> <span class='mono'>("+d.best_move_uci+")</span> — ارزیابی "+(d.eval_cp/100).toFixed(2)+"، شانس برد نوبت "+d.win_percent+"٪<br>خط اصلی: <span class='mono'>"+d.best_line_san.join(" ")+"</span></p>";
  }catch(e){$("fen-result").textContent="خطا: "+e.message;}
}
fetch("/api/health").then(r=>r.json()).then(d=>{$("engine-name").textContent=" | موتور فعال: "+d.engine;}).catch(()=>{});
newGame();
</script></body></html>"""


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
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self):  # noqa: N802 (http.server naming)
            if self.path == "/":
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/health":
                self._send_json({"ok": True, "engine": app.engine.name})
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self):  # noqa: N802
            try:
                data = self._read_json()
                if self.path == "/api/new_game":
                    self._send_json(app.new_game())
                elif self.path == "/api/move":
                    self._send_json(app.human_move(str(data["fen"]), str(data["uci"])))
                elif self.path == "/api/analyze_pgn":
                    self._send_json(app.analyze_pgn_text(str(data["pgn"])))
                elif self.path == "/api/analyze_fen":
                    self._send_json(app.analyze_fen(str(data["fen"])))
                else:
                    self._send_json({"error": "not found"}, 404)
            except (ValueError, KeyError) as exc:
                self._send_json({"error": str(exc)}, 400)
            except Exception as exc:  # pragma: no cover - last-resort guard
                self._send_json({"error": f"خطای سرور: {exc}"}, 500)

    return Handler


def make_server(host: str, port: int, engine) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(_App(engine)))


def serve(host: str = "0.0.0.0", port: int = 8000, engine_arg: str = "auto") -> None:
    engine = get_engine(engine_arg)
    server = make_server(host, port, engine)
    print(f"موتور: {engine.name}")
    print(f"وب‌اپ تمرین: http://{host}:{port}  (Ctrl+C برای خروج)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.close()
