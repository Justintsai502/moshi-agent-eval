"""Load the human-side ground truth and pick out the turns aimed at the agent.

In the real Moshi setup the aligned script contains ONLY the human speakers
(A, B). If it still carries agent rows -- as the scripted TTS sample does --
they are split off and kept as reference answers for calibration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import Utterance, read_jsonl

DEFAULT_WAKE_PATTERNS = (r"\bai\s*agent\b", r"\bhey\s+agent\b", r"\bassistant\b")


@dataclass
class GroundTruth:
    human_turns: list[Utterance]
    agent_turns: list[Utterance]     # empty in a real Moshi run
    queries: list[Utterance]

    @property
    def has_reference(self) -> bool:
        return bool(self.agent_turns)


def _is_query(
    utt: Utterance,
    agent_speaker: str,
    selector: str,
    wake_patterns: tuple[str, ...],
) -> bool:
    if utt.function != "speech":
        return False                     # backchannels are never queries
    if not utt.text.strip():
        return False

    by_addressing = agent_speaker in utt.addressing
    by_wakeword = any(re.search(p, utt.text, re.IGNORECASE) for p in wake_patterns)

    if selector == "addressing":
        return by_addressing
    if selector == "wakeword":
        return by_wakeword
    if selector == "both":
        return by_addressing and by_wakeword
    if selector == "any":
        return by_addressing or by_wakeword
    raise ValueError(f"unknown query selector: {selector!r}")


def load_ground_truth(
    path: str,
    agent_speaker: str = "C",
    query_selector: str = "addressing",
    wake_patterns: tuple[str, ...] = DEFAULT_WAKE_PATTERNS,
) -> GroundTruth:
    """Read aligned_script.jsonl.

    Rows are sorted by ``start`` -- the file is NOT guaranteed to be in time
    order (overlapping turns show up out of sequence), so never trust line
    order here.
    """
    rows = [Utterance.from_gold(r) for r in read_jsonl(path)]
    rows.sort(key=lambda u: (u.start, u.utt_id))

    human = [u for u in rows if u.speaker != agent_speaker]
    agent = [u for u in rows if u.speaker == agent_speaker]
    queries = [
        u for u in human
        if _is_query(u, agent_speaker, query_selector, wake_patterns)
    ]
    return GroundTruth(human_turns=human, agent_turns=agent, queries=queries)


def render_context(
    turns: list[Utterance],
    up_to: float,
    window_s: float = 30.0,
    max_turns: int = 8,
) -> list[str]:
    """Human dialogue lines shortly before ``up_to``, for judge context."""
    picked = [
        u for u in turns
        if u.end <= up_to + 1e-6 and u.start >= up_to - window_s
    ]
    lines = [
        f"[{u.start:7.2f}s] Speaker {u.speaker}"
        f"{' (backchannel)' if u.function == 'backchannel' else ''}: {u.text}"
        for u in picked[-max_turns:]
    ]
    return lines
