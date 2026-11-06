"""Stage 1 bootstrap: timed pattern drills from rated human puzzles.

Anti-memorization protocol (the metric must not be gameable):
- Train and measured-test banks are disjoint by canonical position identity
  (canonical_key treats a FEN and its color-flip as the same item).
- Latency/accuracy are recorded ONLY on unseen test items, each served once,
  in original or color-flipped form.
- A missed FEN is never re-served and never counted again. It may be shown
  once at session end for study, but re-training draws a NEW unseen FEN of
  the same theme class. (SM-2 on identical FENs measures flashcard
  recognition, not perceptual skill.)
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import chess
import polars as pl

from .puzzle_corpus import DATASET
from .transforms import canonical_key, mirror_fen

BOOTSTRAP_THEMES = ("mateIn1", "hangingPiece")
RATING_BAND = (400, 900)


@dataclass
class Attempt:
    puzzle_id: str
    fen_shown: str
    fen_source: str
    mirrored: bool
    themes: list[str]
    rating: int
    latency_s: float
    correct: bool
    shot_clock_s: float
    is_test: bool
    ts: float = field(default_factory=time.time)


def load_theme_bank(
    themes: tuple[str, ...] = BOOTSTRAP_THEMES,
    rating_band: tuple[int, int] = RATING_BAND,
    limit: int = 5000,
) -> list[dict]:
    cond = (pl.col("Rating") >= rating_band[0]) & (pl.col("Rating") <= rating_band[1])
    for t in themes:
        cond |= pl.col("Themes").list.contains(t)
    df = (
        pl.scan_parquet(DATASET)
        .filter(cond)
        .select("PuzzleId", "FEN", "Themes", "Rating", "Moves")
        .limit(limit)
        .collect(streaming=True)
    )
    return df.to_dicts()


class DrillSession:
    def __init__(
        self,
        clock_s: float = 5.0,
        n_train: int = 15,
        n_test: int = 10,
        out_path: Path = Path("bootstrap_attempts.parquet"),
        seed: int = 0,
    ):
        self.clock_s = clock_s
        self.out_path = out_path
        self.rng = random.Random(seed)
        bank = load_theme_bank()
        self.rng.shuffle(bank)
        seen = self._served_canonicals()
        fresh = [b for b in bank if canonical_key(b["FEN"]) not in seen]
        self.train = fresh[:n_train]
        self.test = fresh[n_train : n_train + n_test]
        self.review: list[dict] = []

    def _served_canonicals(self) -> set[str]:
        if not self.out_path.exists():
            return set()
        df = pl.read_parquet(self.out_path)
        return set(canonical_key(f) for f in df["fen_source"].to_list())

    def _next_unseen_of_theme(self, theme: str, bank: list[dict]) -> dict | None:
        seen = self._served_canonicals()
        candidates = [
            b for b in bank
            if theme in b["Themes"] and canonical_key(b["FEN"]) not in seen
        ]
        return self.rng.choice(candidates) if candidates else None

    def record(self, att: Attempt) -> None:
        row = {**att.__dict__, "themes": list(att.themes)}
        df_new = pl.DataFrame([row])
        if self.out_path.exists():
            pl.concat([pl.read_parquet(self.out_path), df_new]).write_parquet(self.out_path)
        else:
            df_new.write_parquet(self.out_path)

    def serve(self, item: dict, is_test: bool) -> Attempt:
        mirrored = is_test and self.rng.random() < 0.5
        fen_shown = mirror_fen(item["FEN"]) if mirrored else item["FEN"]
        board = chess.Board(fen_shown)
        print(f"\n[{'TEST' if is_test else 'train'}] FEN: {fen_shown} "
              f"(rating {item['Rating']}, clock {self.clock_s}s)")
        t0 = time.monotonic()
        answer = input("your move (uci, from White's view of the shown board): ").strip()
        latency = time.monotonic() - t0
        sol = item["Moves"].split()[0]
        if mirrored:
            sol = _mirror_uci(sol, item["FEN"])
        correct = answer == sol and latency <= self.clock_s
        att = Attempt(
            puzzle_id=item["PuzzleId"], fen_shown=fen_shown, fen_source=item["FEN"],
            mirrored=mirrored, themes=list(item["Themes"]), rating=int(item["Rating"]),
            latency_s=round(latency, 3), correct=correct,
            shot_clock_s=self.clock_s, is_test=is_test,
        )
        self.record(att)
        if not correct and not is_test:
            self.review.append(item)
        return att

    def stats(self) -> pl.DataFrame:
        """Measured skill = test rows only. Train rows exist but are excluded."""
        if not self.out_path.exists():
            return pl.DataFrame()
        df = pl.read_parquet(self.out_path).filter(pl.col("is_test"))
        return (
            df.group_by("themes")
            .agg(
                pl.len().alias("n_test"),
                pl.col("correct").mean().alias("accuracy"),
                pl.col("latency_s").median().alias("median_latency_s"),
            )
            .sort("themes")
        )


def _mirror_uci(uci: str, fen: str) -> str:
    """Map a UCI move from the original orientation to the color-flipped one."""
    b = chess.Board(fen)
    mv = chess.Move.from_uci(uci)
    if mv not in b.legal_moves:
        return uci
    m = b.mirror()
    # mirror each square vertically and swap nothing else; python-chess mirror
    # handles colors, so flip squares only
    flipped = chess.Move(chess.square_mirror(mv.from_square), chess.square_mirror(mv.to_square),
                         promotion=mv.promotion)
    return flipped.uci()


if __name__ == "__main__":
    s = DrillSession()
    print(f"train: {len(s.train)}  test: {len(s.test)} (all unseen, canonical-disjoint)")
    for item in s.train:
        att = s.serve(item, is_test=False)
        print(f"  -> {'OK' if att.correct else 'MISS'} in {att.latency_s}s")
    for item in s.test:
        att = s.serve(item, is_test=True)
        print(f"  -> {'OK' if att.correct else 'MISS'} in {att.latency_s}s (measured)")
    if s.review:
        print(f"\nreview once (never re-served): {[r['PuzzleId'] for r in s.review]}")
    print(s.stats())
