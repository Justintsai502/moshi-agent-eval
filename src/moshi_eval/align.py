"""Put authored turns onto the audio clock by aligning them to ASR segments.

The authored script has exact text but no timestamps. `input.wav` has the
timeline but only ASR-quality text. Aligning the two gives each authored turn
(and so each question) a start/end on the clip's clock -- which is the clock
`response.wav` also uses.

Monotonic global alignment (Needleman-Wunsch with free skips on both sides),
scored by token F1. Skips are free because ASR routinely drops a backchannel
and occasionally splits one turn into two segments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def token_f1(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    from collections import Counter
    ca, cb = Counter(ta), Counter(tb)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    p, r = overlap / len(tb), overlap / len(ta)
    return 2 * p * r / (p + r)


@dataclass
class Alignment:
    """turn_index -> (asr_index, score); unmatched turns are absent."""

    matched: dict[int, tuple[int, float]]
    unmatched_turns: list[int]
    unmatched_asr: list[int]

    @property
    def coverage(self) -> float:
        total = len(self.matched) + len(self.unmatched_turns)
        return len(self.matched) / total if total else 0.0


def align_turns_to_asr(
    turn_texts: list[str],
    asr_texts: list[str],
    min_score: float = 0.34,
) -> Alignment:
    n, m = len(turn_texts), len(asr_texts)
    if n == 0 or m == 0:
        return Alignment({}, list(range(n)), list(range(m)))

    sim = [[token_f1(turn_texts[i], asr_texts[j]) for j in range(m)] for i in range(n)]

    # dp[i][j] = best score aligning first i turns with first j asr segments
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    bt = [[0] * (m + 1) for _ in range(n + 1)]   # 0=skip turn, 1=skip asr, 2=match
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            skip_turn = dp[i - 1][j]
            skip_asr = dp[i][j - 1]
            match = dp[i - 1][j - 1] + sim[i - 1][j - 1]
            best = max(skip_turn, skip_asr, match)
            dp[i][j] = best
            bt[i][j] = 2 if best == match else (0 if best == skip_turn else 1)

    matched: dict[int, tuple[int, float]] = {}
    i, j = n, m
    while i > 0 and j > 0:
        move = bt[i][j]
        if move == 2:
            s = sim[i - 1][j - 1]
            if s >= min_score:
                matched[i - 1] = (j - 1, s)
            i, j = i - 1, j - 1
        elif move == 0:
            i -= 1
        else:
            j -= 1

    used_asr = {a for a, _ in matched.values()}
    return Alignment(
        matched=matched,
        unmatched_turns=[i for i in range(n) if i not in matched],
        unmatched_asr=[j for j in range(m) if j not in used_asr],
    )
