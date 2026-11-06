"""UCI client for Maia-3 (CSSLab/maia3): rating-conditioned human move
distributions via SelfElo / OppoElo / MultiPV options."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field

MULTIPV_RE = re.compile(
    r"info .*multipv (?P<rank>\d+) .*score cp (?P<cp>-?\d+).*pv (?P<pv>(?:[a-h][1-8][a-h][1-8][qrbn]?\s*)+)"
)


@dataclass
class MaiaMove:
    rank: int
    cp: int
    pv: list[str]


@dataclass
class MaiaAnalysis:
    fen: str
    self_elo: int
    oppo_elo: int
    moves: list[MaiaMove] = field(default_factory=list)
    bestmove: str | None = None


class MaiaClient:
    def __init__(self, engine_cmd=None, self_elo=1500, oppo_elo=1500, multipv=5):
        self.cmd = engine_cmd or ["maia3-5m"]
        self.proc = subprocess.Popen(
            self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._send("uci"); self._wait_for("uciok")
        self._send(f"setoption name SelfElo value {self_elo}")
        self._send(f"setoption name OppoElo value {oppo_elo}")
        self._send(f"setoption name MultiPV value {multipv}")
        self._send("isready"); self._wait_for("readyok")

    def _send(self, line): self.proc.stdin.write(line + "\n"); self.proc.stdin.flush()
    def _readline(self): return self.proc.stdout.readline().strip()
    def _wait_for(self, token):
        while True:
            line = self._readline()
            if not line: raise RuntimeError("engine terminated")
            if line.startswith(token): return

    def analyze(self, fen, depth=10) -> MaiaAnalysis:
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")
        result = MaiaAnalysis(fen, 0, 0)
        while True:
            line = self._readline()
            if not line: raise RuntimeError("engine terminated during go")
            m = MULTIPV_RE.search(line)
            if m:
                result.moves.append(MaiaMove(int(m.group("rank")), int(m.group("cp")), m.group("pv").split()))
            if line.startswith("bestmove"):
                result.bestmove = line.split()[1]; break
        result.moves.sort(key=lambda mv: mv.rank)
        return result

    def close(self):
        self._send("quit"); self.proc.wait(timeout=10)
