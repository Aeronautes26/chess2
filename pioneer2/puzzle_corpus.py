"""Remote Polars queries over the Lichess puzzle corpus."""

import polars as pl

DATASET = "hf://datasets/Lichess/chess-puzzles-with-games/data/train-*.parquet"


def themed_puzzles(
    themes: tuple[str, ...] | None = None,
    any_theme: bool = True,
    min_rating: int = 0,
    limit: int = 10_000,
) -> pl.LazyFrame:
    lf = pl.scan_parquet(DATASET).filter(pl.col("Rating") >= min_rating)
    if themes:
        cond = None
        for t in themes:
            c = pl.col("Themes").list.contains(t)
            cond = c if cond is None else (cond | c if any_theme else cond & c)
        lf = lf.filter(cond)
    return lf.select(
        "PuzzleId", "FEN", "Moves", "Rating", "Themes",
        "WhiteElo", "BlackElo", "perf", "speed", "GameId",
    ).limit(limit)
