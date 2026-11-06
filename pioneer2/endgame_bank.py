"""Real low-density positions, sampled from actual games.

Per de Groot / Chase & Simon: experts have NO advantage over novices on
randomized boards. Chunking is ecological - it lives in real pawn-and-piece
constellations. So sparse training positions must be real endgames
(6-10 pieces) from the game database, never synthetic scatters.

Source: Lichess/chess-puzzles-with-games (every row is a real in-game
position with metadata). Piece count is derived from the FEN board field
inside the remote scan, so only matching rows cross the network.
"""

from __future__ import annotations

import polars as pl

PUZZLES = "hf://datasets/Lichess/chess-puzzles-with-games/data/train-*.parquet"


def _piece_count_expr() -> pl.Expr:
    board_field = pl.col("FEN").str.split(" ").list.get(0)
    return board_field.str.replace_all(r"[1-8]", "").str.len_chars()


def real_endgame_bank(
    min_pieces: int = 6,
    max_pieces: int = 10,
    min_rating: int = 0,
    limit: int = 5000,
) -> pl.LazyFrame:
    return (
        pl.scan_parquet(PUZZLES)
        .with_columns(_piece_count_expr().alias("piece_count"))
        .filter(
            (pl.col("Rating") >= min_rating)
            & (pl.col("piece_count") >= min_pieces)
            & (pl.col("piece_count") <= max_pieces)
        )
        .select("PuzzleId", "FEN", "Themes", "Rating", "piece_count")
        .limit(limit)
    )


if __name__ == "__main__":
    df = real_endgame_bank(limit=10).collect(streaming=True)
    print(df)
