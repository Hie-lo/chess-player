"""Read a chess position from an *image the user manually provides*.

Workflow in the web app (deliberately manual, like board-scanner apps for
books and puzzles):

1. the user drops an image file and marks the board's two opposite
   corners on it in the browser;
2. the browser crops the board square and sends it here;
3. this module splits it into 64 squares and proposes a piece per square
   (background subtraction -> piece colour -> template match on two
   channels: silhouette *and* internal detail lines);
4. the user verifies/corrects the proposal on an editor board;
5. only then does the verified FEN go to the engine for live analysis.

Why two template channels?  Silhouette alone confuses look-alikes (in
many fonts a bishop's outline is a pawn with a ball on top, and a queen's
crown spikes merge into a blob).  The *detail* channel — dark strokes
inside white pieces, light strokes inside black pieces (the bishop's
cross, the queen's crown lines, the knight's eye) — is what humans use
too, so the templates store both, separately per colour.

Limitations, honestly: template matching against a chess font is a
heuristic, not a trained CNN.  It works well on clean digital boards and
so-so on fancy 3-D sets — which is exactly why step 4 (human
verification) exists and why this is a *position import* tool, not a
live overlay.  A small CNN trained on rendered boards (many piece sets /
themes) is the production upgrade path; see docs/ARCHITECTURE.md.

Requires the ``image`` extra:  pip install "chess-player[image]"
(Pillow + NumPy).  Without it, functions raise ImageImportUnavailable.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path

import chess

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - exercised only without extras
    np = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class ImageImportUnavailable(RuntimeError):
    """Raised when Pillow/NumPy are not installed."""


LIGHT_SQUARE = (236, 220, 185)
DARK_SQUARE = (181, 136, 99)
_FILLED_GLYPH = {  # filled silhouettes — same shape for both colours
    chess.KING: "\u265a",
    chess.QUEEN: "\u265b",
    chess.ROOK: "\u265c",
    chess.BISHOP: "\u265d",
    chess.KNIGHT: "\u265e",
    chess.PAWN: "\u265f",
}
#: Template keys are FEN letters — uppercase templates for white pieces
#: (bright fill, dark detail), lowercase for black (dark fill, light detail).
_TEMPLATE_CHARS = list("KQRBNPkqrbnp")
_TEMPLATES_PATH = Path(__file__).with_name("piece_templates.npz")
_TEMPLATE_SIZE = 48

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "C:/Windows/Fonts/seguisym.ttf",
    "C:/Windows/Fonts/arialuni.ttf",
]


def _require_image_stack() -> None:
    if _IMPORT_ERROR is not None:
        raise ImageImportUnavailable(
            'Image import needs Pillow and NumPy: pip install "chess-player[image]"'
        ) from _IMPORT_ERROR


def find_chess_font(size: int) -> "ImageFont.FreeTypeFont":
    _require_image_stack()
    for candidate in [os.environ.get("CHESS_FONT_PATH"), *_FONT_CANDIDATES]:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    raise ImageImportUnavailable(
        "No font with chess glyphs found; set $CHESS_FONT_PATH to a font "
        "like DejaVu Sans (needed only to *render* boards/templates — "
        "recognition itself uses the bundled piece_templates.npz)."
    )


# ---------------------------------------------------------------------------
# Feature extraction: silhouette + internal-detail masks for one square
# ---------------------------------------------------------------------------


@dataclass
class _Features:
    silhouette: "np.ndarray"  # normalised (_TEMPLATE_SIZE^2) bool
    detail: "np.ndarray"  # same crop/scale as silhouette
    is_white: bool
    height_ratio: float  # silhouette bbox height / tile height
    mask_fraction: float  # occupied fraction of the raw tile


def _fit_canvas(masks: list["np.ndarray"], bbox: tuple[int, int, int, int]):
    """Crop all masks to bbox, fit (aspect preserved) centred on canvas."""
    y0, y1, x0, x1 = bbox
    cropped = [m[y0 : y1 + 1, x0 : x1 + 1] for m in masks]
    h, w = cropped[0].shape
    scale = _TEMPLATE_SIZE / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    oy, ox = (_TEMPLATE_SIZE - new_h) // 2, (_TEMPLATE_SIZE - new_w) // 2
    out = []
    for crop in cropped:
        resized = Image.fromarray((crop * 255).astype("uint8")).resize((new_w, new_h))
        canvas = np.zeros((_TEMPLATE_SIZE, _TEMPLATE_SIZE), dtype=bool)
        canvas[oy : oy + new_h, ox : ox + new_w] = np.asarray(resized) > 127
        out.append(canvas)
    return out


def _extract_features(tile: "np.ndarray", force_white: bool | None = None) -> _Features:
    """tile: HxWx3 float RGB array of a single square."""
    h, w, _ = tile.shape
    patch = max(4, h // 8)
    corners = np.concatenate(
        [
            tile[:patch, :patch].reshape(-1, 3),
            tile[:patch, -patch:].reshape(-1, 3),
            tile[-patch:, :patch].reshape(-1, 3),
            tile[-patch:, -patch:].reshape(-1, 3),
        ]
    )
    background = np.median(corners, axis=0)
    distance = np.sqrt(((tile - background) ** 2).sum(axis=2))
    mask = distance > 45
    mask_fraction = float(mask.mean())
    if mask_fraction == 0:
        empty = np.zeros((_TEMPLATE_SIZE, _TEMPLATE_SIZE), dtype=bool)
        return _Features(empty, empty, bool(force_white), 0.0, 0.0)

    luminance = tile.mean(axis=2)
    fg_lum = luminance[mask]
    if force_white is not None:
        is_white = force_white
    else:
        bright = float((fg_lum > 170).mean())
        dark = float((fg_lum < 80).mean())
        is_white = bright > dark
    # Internal detail: outline/detail-colour pixels inside the silhouette.
    detail = mask & (luminance < 80 if is_white else luminance > 170)

    ys, xs = np.nonzero(mask)
    bbox = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
    sil_norm, det_norm = _fit_canvas([mask, detail], bbox)
    return _Features(
        silhouette=sil_norm,
        detail=det_norm,
        is_white=is_white,
        height_ratio=float((bbox[1] - bbox[0] + 1) / h),
        mask_fraction=mask_fraction,
    )


# ---------------------------------------------------------------------------
# Templates: one (silhouette, detail) pair per piece type per colour
# ---------------------------------------------------------------------------


@dataclass
class _Templates:
    masks: dict[str, "np.ndarray"]
    details: dict[str, "np.ndarray"]
    height_ratios: dict[str, float]


def _draw_glyph_tile(piece_type: int, is_white: bool, tile: int = 120) -> "np.ndarray":
    img = Image.new("RGB", (tile, tile), LIGHT_SQUARE)
    draw = ImageDraw.Draw(img)
    font = find_chess_font(int(tile * 0.82))
    glyph = _FILLED_GLYPH[piece_type]
    bbox = draw.textbbox((0, 0), glyph, font=font)
    w_px, h_px = bbox[2] - bbox[0], bbox[3] - bbox[1]
    fill, outline = (
        ((245, 245, 245), (15, 15, 15)) if is_white else ((25, 25, 25), (235, 235, 235))
    )
    draw.text(
        ((tile - w_px) / 2 - bbox[0], (tile - h_px) / 2 - bbox[1]),
        glyph, fill=fill, font=font, stroke_width=2, stroke_fill=outline,
    )
    return np.asarray(img).astype(float)


def build_templates() -> _Templates:
    """Render each piece once per colour and extract template features."""
    _require_image_stack()
    masks: dict[str, "np.ndarray"] = {}
    details: dict[str, "np.ndarray"] = {}
    heights: dict[str, float] = {}
    for is_white in (True, False):
        for piece_type in chess.PIECE_TYPES:
            tile = _draw_glyph_tile(piece_type, is_white)
            feat = _extract_features(tile, force_white=is_white)
            char = chess.piece_symbol(piece_type)
            char = char.upper() if is_white else char.lower()
            masks[char], details[char] = feat.silhouette, feat.detail
            heights[char] = feat.height_ratio
    return _Templates(masks, details, heights)


def save_templates(path: Path = _TEMPLATES_PATH) -> Path:
    templates = build_templates()
    np.savez_compressed(
        path,
        chars=np.array(_TEMPLATE_CHARS),
        masks=np.stack([templates.masks[c] for c in _TEMPLATE_CHARS]),
        details=np.stack([templates.details[c] for c in _TEMPLATE_CHARS]),
        height_ratios=np.array([templates.height_ratios[c] for c in _TEMPLATE_CHARS]),
    )
    return path


_TEMPLATES_CACHE: _Templates | None = None


def get_templates() -> _Templates:
    global _TEMPLATES_CACHE
    _require_image_stack()
    if _TEMPLATES_CACHE is not None:
        return _TEMPLATES_CACHE
    if _TEMPLATES_PATH.exists():
        data = np.load(_TEMPLATES_PATH)
        if "details" in data:  # current format
            chars = [str(c) for c in data["chars"]]
            _TEMPLATES_CACHE = _Templates(
                masks={c: data["masks"][i] for i, c in enumerate(chars)},
                details={c: data["details"][i] for i, c in enumerate(chars)},
                height_ratios={
                    c: float(data["height_ratios"][i]) for i, c in enumerate(chars)
                },
            )
            return _TEMPLATES_CACHE
    _TEMPLATES_CACHE = build_templates()
    return _TEMPLATES_CACHE


# ---------------------------------------------------------------------------
# Rendering (also handy for tests and docs)
# ---------------------------------------------------------------------------


def render_board_image(board: chess.Board, square_size: int = 100) -> bytes:
    """Render a board as a clean digital-board PNG (White at the bottom)."""
    _require_image_stack()
    size = square_size * 8
    img = Image.new("RGB", (size, size), LIGHT_SQUARE)
    draw = ImageDraw.Draw(img)
    for rank in range(8):  # rank 0 = rank1 (bottom)
        for file in range(8):
            x, y = file * square_size, (7 - rank) * square_size
            if (file + rank) % 2 == 0:
                draw.rectangle(
                    [x, y, x + square_size - 1, y + square_size - 1], fill=DARK_SQUARE
                )
    font = find_chess_font(int(square_size * 0.82))
    for square, piece in board.piece_map().items():
        file, rank = chess.square_file(square), chess.square_rank(square)
        x, y = file * square_size, (7 - rank) * square_size
        glyph = _FILLED_GLYPH[piece.piece_type]
        bbox = draw.textbbox((0, 0), glyph, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cx = x + (square_size - w) / 2 - bbox[0]
        cy = y + (square_size - h) / 2 - bbox[1]
        if piece.color == chess.WHITE:
            fill, outline = (245, 245, 245), (15, 15, 15)
        else:
            fill, outline = (25, 25, 25), (235, 235, 235)
        # A single stroked draw keeps internal negative space intact.
        draw.text(
            (cx, cy), glyph, fill=fill, font=font,
            stroke_width=max(1, square_size // 50), stroke_fill=outline,
        )
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------


def _iou(a: "np.ndarray", b: "np.ndarray") -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _classify_tile(tile: "np.ndarray", templates: _Templates) -> tuple[str, float]:
    feat = _extract_features(tile)
    if feat.mask_fraction < 0.035:
        return "", 0.95
    candidates = "KQRBNP" if feat.is_white else "kqrbnp"
    best_char, best_score = "", -1.0
    for char in candidates:
        score = (
            0.58 * _iou(feat.silhouette, templates.masks[char])
            + 0.32 * _iou(feat.detail, templates.details[char])
            + 0.10
            * (1.0 - min(1.0, abs(feat.height_ratio - templates.height_ratios[char])))
        )
        if score > best_score:
            best_char, best_score = char, score
    return best_char, round(float(best_score), 2)


def board_grid_from_image(image_bytes: bytes) -> dict:
    """Propose an 8x8 piece grid from a cropped board image.

    Input should be (roughly) just the board itself — the web UI crops to
    the corners the user marks, and non-square inputs are centre-cropped
    to a square here.  Row 0 is the top of the image (rank 8 when White
    is at the bottom).  Chars are FEN letters: ``PNBRQK`` white,
    ``pnbrqk`` black, ``""`` empty.  Treat the result as a *proposal*:
    the UI lets the user correct squares before any analysis runs.
    """
    _require_image_stack()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    side = min(img.size)
    left, top = (img.size[0] - side) // 2, (img.size[1] - side) // 2
    img = img.crop((left, top, left + side, top + side)).resize((800, 800))
    arr = np.asarray(img).astype(float)
    templates = get_templates()

    grid: list[list[str]] = []
    confidences: list[list[float]] = []
    for row in range(8):
        grid_row: list[str] = []
        conf_row: list[float] = []
        for col in range(8):
            tile = arr[row * 100 : (row + 1) * 100, col * 100 : (col + 1) * 100]
            char, conf = _classify_tile(tile, templates)
            grid_row.append(char)
            conf_row.append(conf)
        grid.append(grid_row)
        confidences.append(conf_row)
    return {"grid": grid, "confidences": confidences}


def fen_from_grid(grid: list[list[str]], turn: str = "white") -> str:
    """Build a FEN from an 8x8 grid (row 0 = rank 8). Castling/rights '-'.

    The user picks the side to move in the UI; clocks default to 0/1.
    Legality is *not* guaranteed for arbitrary grids — callers validate
    with ``chess.Board(fen).is_valid()`` and show the status to the user.
    """
    rows = []
    for row in grid:
        fen_row, empty = "", 0
        for char in row:
            if not char:
                empty += 1
                continue
            if empty:
                fen_row += str(empty)
                empty = 0
            fen_row += char
        if empty:
            fen_row += str(empty)
        rows.append(fen_row)
    return "/".join(rows) + f" {'w' if turn == 'white' else 'b'} - - 0 1"


if __name__ == "__main__":  # regenerate the bundled templates
    print("templates ->", save_templates())
