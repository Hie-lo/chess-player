"""Command line interface.

Examples
--------
Analyse a finished game exported from Chess.com / Lichess::

    python -m chess_player analyze examples/opera_game.pgn --depth 14

Ask the engine about one position you set up yourself (analysis board)::

    python -m chess_player fen "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"

Read a position from a board image you supply, and analyse it::

    python -m chess_player image my_board.png --turn white --analyze

Launch the live-analysis web studio (analysis board + play vs engine
+ guess-the-move training + PGN review + image import)::

    python -m chess_player serve --port 8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess
import chess.pgn

from . import report
from .analysis import analyze_game, analyze_position
from .engine import get_engine


def _add_engine_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--engine",
        default="auto",
        help="'auto' (Stockfish if found), 'simple' (built-in fallback), "
        "or a path to a UCI engine binary, e.g. /usr/bin/stockfish",
    )
    parser.add_argument(
        "--depth", type=int, default=14,
        help="Search depth. Stockfish: 12-18 is a good speed/quality trade-off "
        "for whole games; the fallback engine is capped at 3.",
    )


def cmd_analyze(args: argparse.Namespace) -> int:
    pgn_path = Path(args.pgn)
    if not pgn_path.exists():
        print(f"فایل PGN پیدا نشد: {pgn_path}", file=sys.stderr)
        return 1
    with pgn_path.open(encoding="utf-8", errors="replace") as fh:
        game = chess.pgn.read_game(fh)
    if game is None:
        print("PGN خالی یا نامعتبر است.", file=sys.stderr)
        return 1

    engine = get_engine(args.engine)
    try:
        print(f"موتور: {engine.name}\nدر حال تحلیل {pgn_path.name} ...", file=sys.stderr)
        analysis = analyze_game(engine, game, depth=args.depth)
    finally:
        engine.close()

    if args.json:
        Path(args.json).write_text(report.to_json(analysis), encoding="utf-8")
        print(f"JSON  -> {args.json}", file=sys.stderr)
    if args.html:
        Path(args.html).write_text(report.to_html(analysis), encoding="utf-8")
        print(f"HTML  -> {args.html}", file=sys.stderr)
    if args.out:
        Path(args.out).write_text(report.to_markdown(analysis), encoding="utf-8")
        print(f"MD    -> {args.out}", file=sys.stderr)
    if not (args.json or args.html or args.out):
        print(report.to_markdown(analysis))
    return 0


def cmd_fen(args: argparse.Namespace) -> int:
    try:
        board = chess.Board(args.fen)
    except ValueError as exc:
        print(f"FEN نامعتبر: {exc}", file=sys.stderr)
        return 1
    if not board.is_valid():
        print(f"وضعیت غیرقانونی: {board.status()}", file=sys.stderr)
        return 1
    engine = get_engine(args.engine)
    try:
        info = analyze_position(engine, board, depth=args.depth)
    finally:
        engine.close()
    print(board)
    print(f"\nنوبت: {'سفید' if info['turn'] == 'white' else 'سیاه'}")
    mate = f" (مات در {info['mate']})" if info["mate"] else ""
    print(f"ارزیابی: {info['eval_cp'] / 100:+.2f} پیاده{mate} — شانس برد نوبت: {info['win_percent']}٪")
    print(f"بهترین حرکت: {info['best_move_san']} ({info['best_move_uci']})")
    print(f"خط اصلی: {' '.join(info['best_line_san'])}")
    print(f"موتور: {info['engine']} | عمق {info['depth']}")
    return 0


def cmd_image(args: argparse.Namespace) -> int:
    """Read a position from an image file the user supplies (cropped board).

    Prints the proposed FEN (and, with --analyze, the top engine lines).
    Always eyeball the printed board against your image before trusting
    it — the web app's editor is the safer path for real screenshots.
    """
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"فایل تصویر پیدا نشد: {image_path}", file=sys.stderr)
        return 1
    try:
        from .analysis import analyze_position_multipv
        from .image_fen import ImageImportUnavailable, board_grid_from_image, fen_from_grid
    except ImportError as exc:  # pragma: no cover
        print(f"واردکردن تصویر در دسترس نیست: {exc}", file=sys.stderr)
        return 1
    try:
        proposal = board_grid_from_image(image_path.read_bytes())
    except ImageImportUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 1
    fen = fen_from_grid(proposal["grid"], args.turn)
    board = chess.Board(fen)
    print(board)
    print(f"\nFEN حدسی: {fen}")
    low = [
        (r, c)
        for r, row in enumerate(proposal["confidences"])
        for c, conf in enumerate(row)
        if conf < 0.55 and proposal["grid"][r][c]
    ]
    if low:
        squares = ", ".join(f"{'abcdefgh'[c]}{8 - r}" for r, c in low)
        print(f"⚠️ خانه‌های کم‌اطمینان (حتماً بررسی کنید): {squares}")
    if not board.is_valid():
        print(f"⚠️ وضعیت قانونی نیست (کد {board.status()}) — در وب‌اپ با ویرایشگر اصلاح کنید.")
        return 1
    if args.analyze:
        engine = get_engine(args.engine)
        try:
            info = analyze_position_multipv(engine, board, depth=args.depth, multipv=3)
        finally:
            engine.close()
        print(f"\nموتور: {info['engine']} | عمق {info['depth']} | شانس برد سفید {info['win_percent_white']}٪")
        for line in info["lines"]:
            print(
                f"  {line['rank']}. {line['move_san']} "
                f"({line['eval_white_cp'] / 100:+.2f})  {' '.join(line['pv_san'])}"
            )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .webapp import serve

    serve(host=args.host, port=args.port, engine_arg=args.engine)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chess-player",
        description="تحلیل پس از بازی و تمرین آفلاین شطرنج با Stockfish",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="تحلیل یک بازی تمام‌شده (فایل PGN)")
    p_analyze.add_argument("pgn", help="مسیر فایل PGN")
    p_analyze.add_argument("--out", help="ذخیرهٔ گزارش Markdown در این فایل")
    p_analyze.add_argument("--json", help="ذخیرهٔ گزارش JSON در این فایل")
    p_analyze.add_argument("--html", help="ذخیرهٔ گزارش HTML در این فایل")
    _add_engine_args(p_analyze)
    p_analyze.set_defaults(func=cmd_analyze)

    p_fen = sub.add_parser("fen", help="بهترین حرکت یک وضعیت دستی (FEN) — تختهٔ تحلیل")
    p_fen.add_argument("fen", help="رشتهٔ FEN وضعیت")
    _add_engine_args(p_fen)
    p_fen.set_defaults(func=cmd_fen)

    p_image = sub.add_parser(
        "image", help="خواندن وضعیت از تصویر تخته (فایل دستی، برش‌خورده روی خود تخته)"
    )
    p_image.add_argument("image", help="مسیر فایل تصویر (PNG/JPG) — فقط خود تخته، مربعی")
    p_image.add_argument("--turn", choices=["white", "black"], default="white")
    p_image.add_argument("--analyze", action="store_true", help="نمایش سه خط برتر موتور")
    _add_engine_args(p_image)
    p_image.set_defaults(func=cmd_image)

    p_serve = sub.add_parser("serve", help="اجرای وب‌اپ استودیو تحلیل زنده")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--engine", default="auto")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
