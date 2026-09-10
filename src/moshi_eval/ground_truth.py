"""Load the human-side ground truth and pick out the turns aimed at the agent.

In the real Moshi setup the aligned script contains ONLY the human speakers
(A, B). If it still carries agent rows -- as the scripted TTS sample does --
they are split off and kept as reference answers for calibration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import Utterance, read_jsonl

# Two turns starting within this window are treated as simultaneous: which one
# "came first" is then decided by discourse role, not by sub-frame jitter.
SIMULTANEITY_TOL_S = 0.05

_NUM_RE = re.compile(r"(\d+)")


def natural_key(utt_id: str) -> tuple:
    """Split digits out so g9 sorts before g10, not after."""
    return tuple(
        int(part) if part.isdigit() else part
        for part in _NUM_RE.split(utt_id)
    )


def _role_rank(u: Utterance) -> int:
    """Backchannels render after the speech they are layered onto.

    A backchannel is a reaction on top of someone else's ongoing turn. Placing
    it first makes the transcript read as if the reaction preceded its trigger.
    """
    return 1 if u.function == "backchannel" else 0


def sort_utterances(rows: list[Utterance]) -> list[Utterance]:
    """Order turns by time, resolving near-simultaneous starts sensibly.

    Row order in these files is authoring order, not time order, and utt_id
    order is neither -- in the road-trip sample g006 precedes g005 in the file
    while starting 11.7 ms earlier, and the two overlap for 8 s.

    Pass 1 sorts strictly by (start, end, natural utt_id): deterministic, and
    never depends on how the file happened to be written.

    Pass 2 walks that order and re-sorts each run of turns whose starts fall
    within SIMULTANEITY_TOL_S of the run's first start, by (role, end, id).
    Doing it as a local sweep rather than by quantising `start` avoids the
    boundary artefact where two turns 40 ms apart land in different buckets.

    `gap_ms` is deliberately ignored: the sample's top-level value disagrees
    with `rel.gap_ms` on overlapping turns (g005: -7081.6 vs +77.0). Only
    `start`/`end` are trusted.
    """
    ordered = sorted(rows, key=lambda u: (u.start, u.end, natural_key(u.utt_id)))

    out: list[Utterance] = []
    i = 0
    while i < len(ordered):
        j = i + 1
        anchor = ordered[i].start
        while j < len(ordered) and ordered[j].start - anchor <= SIMULTANEITY_TOL_S:
            j += 1
        group = ordered[i:j]
        if len(group) > 1:
            group.sort(key=lambda u: (_role_rank(u), u.end, natural_key(u.utt_id)))
        out += group
        i = j
    return out


def find_overlaps(rows: list[Utterance]) -> list[tuple[str, str, float]]:
    """(earlier_id, later_id, seconds_of_overlap) for every overlapping pair."""
    ordered = sorted(rows, key=lambda u: u.start)
    hits: list[tuple[str, str, float]] = []
    for a_i, a in enumerate(ordered):
        for b in ordered[a_i + 1:]:
            if b.start >= a.end:
                break
            hits.append((a.utt_id, b.utt_id, min(a.end, b.end) - b.start))
    return hits

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

    Never trust line order: the file is written in authoring order, which for
    overlapping turns is not time order. See ``sort_utterances``.
    """
    rows = sort_utterances([Utterance.from_gold(r) for r in read_jsonl(path)])

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
        u for u in sort_utterances(turns)
        if u.end <= up_to + 1e-6 and u.start >= up_to - window_s
    ]
    lines = [
        f"[{u.start:7.2f}s] Speaker {u.speaker}"
        f"{' (backchannel)' if u.function == 'backchannel' else ''}: {u.text}"
        for u in picked[-max_turns:]
    ]
    return lines
