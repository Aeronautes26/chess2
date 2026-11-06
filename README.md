# chess2 — PIONEER II bootstrap instrument

Personal chess research instrument. Current focus: the beginner bootstrap —
teaching board vision (space) before calculation (time).

## Why this design

- de Groot / Chase & Simon: masters have **zero** visual advantage on random
  boards. Chunking is ecological (real pawn/piece constellations), so all
  low-density training positions are **real 6–10 piece game positions**
  (`endgame_bank.py`), never synthetic scatters.
- Beginner misses are backward-scan / change-blindness failures, not
  "can't plot a knight's squares." So queries are **inverted**: never
  "where does this highlighted piece go" — instead "which enemy pieces
  attack or x-ray your king?" (`vision_gym.py`). X-ray/occlusion (slider
  blocked by one interposed piece) is trained explicitly.
- The evaluation metric must not be gameable: latency/accuracy are measured
  **only on unseen, color-flipped test FENs, each served once**
  (`bootstrap_drill.py`, `transforms.py`). Missed FENs are never re-served;
  re-training draws a new unseen FEN of the same theme. No SM-2 on identical
  positions — that measures flashcard memory, not vision.

## Modules

- `endgame_bank.py` — real 6–10 piece positions via remote Polars scan.
- `vision_gym.py` — stage 0: king-scan / target-reach drills with x-ray answers.
- `bootstrap_drill.py` — stage 1: timed rated mateIn1/hangingPiece drills
  (400–900), disjoint train/test, measured-on-test-only telemetry.
- `puzzle_corpus.py` — themed remote queries over Lichess puzzle parquet.
- `maia_client.py` — Maia-3 UCI wrapper (human move distributions by rating).
- `transforms.py` — color-flip mirroring + canonical identity for anti-memorization.

## Run

```bash
pip install -r requirements.txt
python -m pioneer2.vision_gym      # stage 0 (needs network for the bank)
python -m pioneer2.bootstrap_drill # stage 1 CLI drill
```
