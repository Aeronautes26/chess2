"""ASCII board rendering for CLI drills."""

from __future__ import annotations

import chess

SYMBOLS = {
    chess.PAWN: "p", chess.KNIGHT: "n", chess.BISHOP: "b",
    chess.ROOK: "r", chess.QUEEN: "q", chess.KING: "k",
}


def render_board(fen: str, highlight: set[int] | None = None) -> str:
    board = chess.Board(fen)
    highlight = highlight or set()
    lines = []
    for rank in range(7, -1, -1):
        row = []
        for file in range(8):
            sq = chess.square(file, rank)
            pc = board.piece_at(sq)
            ch = SYMBOLS[pc.piece_type].upper() if pc and pc.color == chess.WHITE else SYMBOLS[pc.piece_type] if pc else "."
            row.append(f"[{ch}]" if sq in highlight else f" {ch} ")
        lines.append(f"{rank + 1} " + "".join(row))
    lines.append("   a  b  c  d  e  f  g  h")
    return "\n".join(lines)
