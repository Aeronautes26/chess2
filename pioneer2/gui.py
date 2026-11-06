"""Tkinter GUI for the bootstrap drills. No new dependencies.

Modes:
  vision  - stage 0: click the squares that answer the inverted query
            ("which enemy pieces attack/x-ray your king?"), instant grading.
  puzzle  - stage 1: timed mateIn1/hangingPiece puzzles, click from-square
            then to-square. Test items are unseen + color-flipped, served once.

All attempts are appended to bootstrap_attempts.parquet with the same schema
as the CLI, so telemetry is identical either way.
"""

from __future__ import annotations

import random
import time
import tkinter as tk
from pathlib import Path

import chess

from .board_view import SYMBOLS
from .bootstrap_drill import Attempt, DrillSession, load_theme_bank
from .transforms import canonical_key, mirror_fen
from .vision_gym import DrillItem, king_scan_item, make_session, target_reach_item

GLYPH = {
    (chess.PAWN, True): "♙", (chess.KNIGHT, True): "♘", (chess.BISHOP, True): "♗",
    (chess.ROOK, True): "♖", (chess.QUEEN, True): "♕", (chess.KING, True): "♔",
    (chess.PAWN, False): "♟", (chess.KNIGHT, False): "♞", (chess.BISHOP, False): "♝",
    (chess.ROOK, False): "♜", (chess.QUEEN, False): "♛", (chess.KING, False): "♚",
}
LIGHT, DARK = "#f0d9b5", "#b58863"
GOOD, BAD, MISS = "#7bc47f", "#e06c6c", "#e8d44d"
SQ = 64


class BoardCanvas(tk.Canvas):
    def __init__(self, master, on_square_click):
        super().__init__(master, width=8 * SQ, height=8 * SQ, highlightthickness=0)
        self.on_square_click = on_square_click
        self.board: chess.Board | None = None
        self.selected: set[int] = set()
        self.marks: dict[int, str] = {}
        self.click_enabled = True
        self.bind("<Button-1>", self._click)

    def _click(self, ev):
        if not self.click_enabled or not self.board:
            return
        file, rank = ev.x // SQ, 7 - ev.y // SQ
        self.on_square_click(chess.square(file, rank))

    def draw(self, fen: str):
        self.board = chess.Board(fen)
        self.selected.clear()
        self.marks.clear()
        self._render()

    def mark(self, squares: dict[int, str]):
        self.marks = squares
        self._render()

    def select(self, squares: set[int]):
        self.selected = set(squares)
        self._render()

    def _render(self):
        self.delete("all")
        for rank in range(8):
            for file in range(8):
                sq = chess.square(file, 7 - rank)
                x, y = file * SQ, rank * SQ
                color = LIGHT if (file + rank) % 2 == 0 else DARK
                if sq in self.marks:
                    color = self.marks[sq]
                elif sq in self.selected:
                    color = "#4d9de0"
                self.create_rectangle(x, y, x + SQ, y + SQ, fill=color, outline="")
                pc = self.board.piece_at(sq)
                if pc:
                    self.create_text(
                        x + SQ // 2, y + SQ // 2, text=GLYPH[(pc.piece_type, pc.color)],
                        font=("DejaVu Sans", SQ // 2), fill="#111111" if pc.color else "#000000",
                    )
        for rank in range(8):
            self.create_text(6, rank * SQ + SQ // 2, text=str(8 - rank), font=("DejaVu Sans", 10), fill="#555555")
        for file in range(8):
            self.create_text(file * SQ + SQ - 10, 8 * SQ - 8, text="abcdefgh"[file], font=("DejaVu Sans", 10), fill="#555555")


class DrillApp(tk.Tk):
    def __init__(self, mode: str = "vision", seed: int = 0):
        super().__init__()
        self.title(f"chess2 bootstrap — {mode}")
        self.mode = mode
        self.rng = random.Random(seed)
        self.log_path = Path("bootstrap_attempts.parquet")
        self.status = tk.Label(self, text="loading bank…", font=("DejaVu Sans", 12))
        self.status.pack()
        self.canvas = BoardCanvas(self, self._on_square)
        self.canvas.pack()
        self.btn = tk.Button(self, text="Submit", command=self._submit)
        self.btn.pack()
        self.clock_label = tk.Label(self, text="", font=("DejaVu Sans", 12))
        self.clock_label.pack()
        self.after(50, self._start)

    def _start(self):
        if self.mode == "vision":
            from .vision_gym import load_bank
            self.items = make_session(load_bank(limit=500), n_items=20, seed=self.rng.randint(0, 10**9))
        else:
            bank = load_theme_bank(limit=2000)
            self.rng.shuffle(bank)
            self.items = bank
        self.idx = 0
        self._next()

    def _next(self):
        if self.idx >= len(self.items):
            self.status.config(text="session done — see bootstrap_attempts.parquet")
            self.canvas.click_enabled = False
            self.btn.config(state="disabled")
            return
        self.item = self.items[self.idx]
        self.t0 = time.monotonic()
        if self.mode == "vision":
            self.canvas.draw(self.item.fen)
            self.status.config(text=self.item.question)
        else:
            mirrored = self.rng.random() < 0.5
            self.mirrored = mirrored
            self.fen_shown = mirror_fen(self.item["FEN"]) if mirrored else self.item["FEN"]
            self.canvas.draw(self.fen_shown)
            self.status.config(text="find the move (click from-square, then to-square)")
            self._pending_from = None
        self._tick_clock()

    def _tick_clock(self):
        if self.mode == "puzzle":
            elapsed = time.monotonic() - self.t0
            self.clock_label.config(text=f"{max(0.0, 5.0 - elapsed):.1f}s")
            self.after(100, self._tick_clock)

    def _on_square(self, sq: int):
        if self.mode == "vision":
            if sq in self.canvas.selected:
                self.canvas.selected.discard(sq)
            else:
                self.canvas.selected.add(sq)
            self.canvas.select(self.canvas.selected)
        else:
            if self._pending_from is None:
                self._pending_from = sq
                self.canvas.select({sq})
            else:
                mv = chess.Move(self._pending_from, sq)
                self._pending_from = None
                self._answer_puzzle(mv.uci())

    def _submit(self):
        if self.mode != "vision":
            return
        latency = time.monotonic() - self.t0
        correct = self.canvas.selected == self.item.answer
        marks = {s: GOOD for s in self.item.answer}
        marks.update({s: BAD for s in self.canvas.selected - self.item.answer})
        marks.update({s: MISS for s in self.item.answer - self.canvas.selected})
        self.canvas.mark(marks)
        self.canvas.click_enabled = False
        self._log_vision(latency, correct)
        self.status.config(text=f"{'OK' if correct else 'MISS'} in {latency:.2f}s — next in 1.5s")
        self.after(1500, self._advance)

    def _answer_puzzle(self, uci: str):
        latency = time.monotonic() - self.t0
        sol = self.item["Moves"].split()[0]
        if self.mirrored:
            from .bootstrap_drill import _mirror_uci
            sol = _mirror_uci(sol, self.item["FEN"])
        correct = uci == sol and latency <= 5.0
        self.canvas.click_enabled = False
        self._log_puzzle(latency, correct)
        self.status.config(text=f"{'OK' if correct else 'MISS'} in {latency:.2f}s (solution: {sol}) — next in 1.5s")
        self.after(1500, self._advance)

    def _advance(self):
        self.idx += 1
        self.canvas.click_enabled = True
        self._next()

    def _log_vision(self, latency: float, correct: bool):
        import polars as pl
        row = {
            "puzzle_id": f"vision-{self.idx}", "fen_shown": self.item.fen, "fen_source": self.item.fen,
            "mirrored": False, "themes": [self.item.kind], "rating": 0,
            "latency_s": round(latency, 3), "correct": correct, "shot_clock_s": 0.0,
            "is_test": True, "ts": time.time(),
        }
        self._append(row)

    def _log_puzzle(self, latency: float, correct: bool):
        row = {
            "puzzle_id": self.item["PuzzleId"], "fen_shown": self.fen_shown, "fen_source": self.item["FEN"],
            "mirrored": self.mirrored, "themes": list(self.item["Themes"]), "rating": int(self.item["Rating"]),
            "latency_s": round(latency, 3), "correct": correct, "shot_clock_s": 5.0,
            "is_test": True, "ts": time.time(),
        }
        self._append(row)

    def _append(self, row: dict):
        import polars as pl
        df_new = pl.DataFrame([row])
        if self.log_path.exists():
            pl.concat([pl.read_parquet(self.log_path), df_new]).write_parquet(self.log_path)
        else:
            df_new.write_parquet(self.log_path)


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "vision"
    DrillApp(mode=mode).mainloop()
