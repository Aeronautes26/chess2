"""Position transforms that defeat visual memorization of a specific FEN.

Latency/accuracy may only be measured on unseen, transformed positions.
Color-flipping (vertical mirror + color swap) preserves all geometric
relationships a vision skill should transfer across, while making the exact
pixel pattern novel.
"""

from __future__ import annotations

import chess


def mirror_fen(fen: str) -> str:
    """Color-flipped, vertically mirrored version of a position (legal)."""
    return chess.Board(fen).mirror().fen()


def variants(fen: str) -> list[str]:
    """Canonical test variants: original plus color-flipped mirror."""
    return [fen, mirror_fen(fen)]


def canonical_key(fen: str) -> str:
    """Key that treats a position and its color-flip as the same item,
    so neither is ever served twice in the measured set."""
    return min(fen, mirror_fen(fen))
