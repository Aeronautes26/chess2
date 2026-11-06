"""Stage 0 bootstrap: board vision on REAL sparse positions, inverted queries.

Design constraints (each backed by evidence, not vibes):

1. No random synthetic scatters. Chase & Simon / de Groot: masters show zero
   advantage over novices on randomized boards. Chunking is ecological, so
   low-density items come from real 6-10 piece game positions (endgame_bank).

2. The query is inverted. "Click where this highlighted piece goes" reveals
   the target and teaches nothing about scanning. Real failures are backward
   scan failures and change blindness: attention anchored on the king, missing
   a quiet bishop on a2. So we ask: "which enemy pieces attack (or x-ray)
   your king?" / "which of your pieces can reach this target square?"

3. Occlusion/x-ray is trained explicitly: a slider whose ray is blocked by a
   single interposed piece still "attacks" once that piece moves (e.g. a pawn
   about to capture). Direct attackers alone miss exactly these.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import chess

from .endgame_bank import real_endgame_bank
from .transforms import canonical_key


@dataclass
class DrillItem:
    fen: str
    kind: str
    question: str
    answer: set  # set of squares (chess.Square ints)
    piece_count: int
    canonical: str  # memorization-proof identity of the underlying position


def direct_attackers(board: chess.Board, color: bool, target: chess.Square) -> set:
    return set(chess.SquareSet(board.attackers(color, target)))


def xray_attackers(board: chess.Board, color: bool, target: chess.Square) -> set:
    """Sliders of `color` whose ray to `target` is blocked by exactly one piece
    (e.g. a pawn about to capture off the ray). Knights/pawns/kings have no rays."""
    direct = direct_attackers(board, color, target)
    out = set()
    for sq in chess.SquareSet(board.occupied_co[color]):
        pc = board.piece_at(sq)
        if pc is None or pc.piece_type in (chess.KNIGHT, chess.PAWN, chess.KING):
            continue
        if not chess.ray(sq, target):
            continue
        df = abs(chess.square_file(sq) - chess.square_file(target))
        dr = abs(chess.square_rank(sq) - chess.square_rank(target))
        diagonal = df == dr
        if pc.piece_type == chess.BISHOP and not diagonal:
            continue
        if pc.piece_type == chess.ROOK and diagonal:
            continue
        blockers = chess.popcount(board.occupied & chess.between(sq, target))
        if blockers == 1 and sq not in direct:
            out.add(sq)
    return out


def king_scan_item(board: chess.Board) -> DrillItem:
    """Backward scan from the king anchor: every enemy piece that attacks or
    x-rays the side-to-move's king."""
    king = board.king(board.turn)
    enemy = not board.turn
    attackers = direct_attackers(board, enemy, king) | xray_attackers(board, enemy, king)
    return DrillItem(
        fen=board.fen(),
        kind="king_scan",
        question="select every enemy piece that attacks or x-rays your king",
        answer=attackers,
        piece_count=chess.popcount(board.occupied),
        canonical=canonical_key(board.fen()),
    )


def target_reach_item(board: chess.Board, rng: random.Random) -> DrillItem:
    """Inverted reach query: pick a non-king target square, ask which of the
    side-to-move's pieces can reach it (directly or by x-ray)."""
    own = chess.SquareSet(board.occupied_co[board.turn])
    targets = [s for s in chess.SQUARES if board.piece_at(s) is None or board.piece_at(s).color != board.turn]
    target = rng.choice(targets)
    reach = direct_attackers(board, board.turn, target) | xray_attackers(board, board.turn, target)
    return DrillItem(
        fen=board.fen(),
        kind="target_reach",
        question=f"select every one of your pieces that can reach {chess.square_name(target)}",
        answer=reach,
        piece_count=chess.popcount(board.occupied),
        canonical=canonical_key(board.fen()),
    )


def make_session(
    bank_fens: list[str],
    n_items: int = 20,
    seed: int = 0,
) -> list[DrillItem]:
    rng = random.Random(seed)
    fens = rng.sample(bank_fens, min(n_items, len(bank_fens)))
    items = []
    for i, fen in enumerate(fens):
        board = chess.Board(fen)
        items.append(king_scan_item(board) if i % 2 == 0 else target_reach_item(board, rng))
    return items


def grade(item: DrillItem, response: set) -> bool:
    return set(response) == item.answer


def load_bank(min_pieces: int = 6, max_pieces: int = 10, limit: int = 500) -> list[str]:
    df = real_endgame_bank(min_pieces, max_pieces, limit=limit).collect(streaming=True)
    return df["FEN"].to_list()


if __name__ == "__main__":
    bank = load_bank(limit=50)
    for item in make_session(bank, n_items=6, seed=1):
        b = chess.Board(item.fen)
        print(item.kind, item.piece_count, item.fen)
        print("  ", item.question, "->", sorted(chess.square_name(s) for s in item.answer))
        print("   direct:", sorted(chess.square_name(s) for s in direct_attackers(b, not b.turn, b.king(b.turn))),
              "xray:", sorted(chess.square_name(s) for s in xray_attackers(b, not b.turn, b.king(b.turn))))
