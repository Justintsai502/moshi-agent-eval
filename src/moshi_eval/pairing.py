"""Match each human query to the agent's reply on the ASR'd track.

Both timelines come from the same session clock (the stems are time-aligned
with the mixdown), so start/end values are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ground_truth import GroundTruth, render_context
from .schema import QAPair, Utterance


@dataclass
class PairingConfig:
    # An agent segment counts as a reply if it starts no earlier than
    # query_end - overlap_tolerance_s (barge-in) and within response_window_s.
    overlap_tolerance_s: float = 1.0
    response_window_s: float = 12.0
    # Consecutive agent segments closer than this are one answer.
    merge_gap_s: float = 0.8
    context_window_s: float = 30.0
    context_max_turns: int = 8
    # Bursts with fewer words than this are treated as noise, not answers.
    min_response_words: int = 2
    mode: str = "time"                # time | order
    align_min_score: float = 0.34     # token-F1 floor for authored<->ASR match


def _merge(segs: list[Utterance], gap: float) -> list[list[Utterance]]:
    groups: list[list[Utterance]] = []
    for s in segs:
        if groups and s.start - groups[-1][-1].end <= gap:
            groups[-1].append(s)
        else:
            groups.append([s])
    return groups


def build_pairs(
    gt: GroundTruth,
    asr_segments: list[Utterance],
    cfg: PairingConfig,
    include_reference: bool = True,
) -> list[QAPair]:
    queries = sorted(gt.queries, key=lambda u: u.start)
    groups = _merge(sorted(asr_segments, key=lambda u: u.start), cfg.merge_gap_s)
    used: set[int] = set()
    used_refs: set[str] = set()

    pairs: list[QAPair] = []
    for qi, q in enumerate(queries):
        # A reply must land before the next query is even asked.
        next_q_start = queries[qi + 1].start if qi + 1 < len(queries) else float("inf")
        lo = q.end - cfg.overlap_tolerance_s
        hi = min(q.end + cfg.response_window_s, next_q_start + cfg.overlap_tolerance_s)

        chosen: list[Utterance] | None = None
        for gi, grp in enumerate(groups):
            if gi in used:
                continue
            if lo <= grp[0].start <= hi:
                chosen = grp
                used.add(gi)
                break

        pair = QAPair(
            pair_id=f"p{qi + 1:03d}",
            query_utt_id=q.utt_id,
            query_speaker=q.speaker,
            query_text=q.text,
            query_start=q.start,
            query_end=q.end,
            context=render_context(
                gt.human_turns, q.start, cfg.context_window_s, cfg.context_max_turns
            ),
        )

        if chosen:
            text = " ".join(s.text.strip() for s in chosen).strip()
            pair.response_text = text
            pair.response_start = chosen[0].start
            pair.response_end = chosen[-1].end
            pair.response_utt_ids = [s.utt_id for s in chosen]
            pair.status = "answered" if text else "empty_asr"
        else:
            pair.status = "no_response"

        if include_reference and gt.has_reference:
            pair.reference_text = _nearest_reference(
                gt.agent_turns, lo, hi, used_refs
            )

        pairs.append(pair)

    return pairs


def _nearest_reference(
    agent_turns: list[Utterance],
    lo: float,
    hi: float,
    used_refs: set[str],
) -> str | None:
    """Scripted answer inside the same window used to match the ASR response.

    ``hi`` is already clamped to the next query, so a later question's answer
    cannot leak into this one. Each gold turn is consumed at most once.
    """
    hits = [
        u for u in agent_turns
        if u.utt_id not in used_refs and lo <= u.start <= hi
    ]
    if not hits:
        return None
    used_refs.update(u.utt_id for u in hits)
    return " ".join(u.text for u in hits).strip() or None


def unmatched_segments(
    asr_segments: list[Utterance], pairs: list[QAPair]
) -> list[Utterance]:
    """Agent speech that answered no query -- unprompted output, or a pairing miss."""
    claimed = {uid for p in pairs for uid in p.response_utt_ids}
    return [s for s in asr_segments if s.utt_id not in claimed]


# --- order-based pairing (authored scripts, which have no timestamps) --------

def build_pairs_by_order(
    script,                      # authored.AuthoredScript
    asr_segments: list[Utterance],
    cfg: PairingConfig,
    include_reference: bool = True,
) -> tuple[list[QAPair], list[Utterance]]:
    """Match the k-th agent burst to the k-th question in script order.

    The authored script carries ordering but no clock, so time-based matching
    is impossible here. The agent track is near-silent between answers, so
    burst count normally equals question count; when it does not, every pair
    in the clip is flagged ``count_mismatch`` rather than silently trusted.

    Returns (pairs, unmatched_segments).
    """
    queries = script.queries
    groups = _merge(sorted(asr_segments, key=lambda u: u.start), cfg.merge_gap_s)

    # Drop blips too short to be an answer (breath, click, a stray token).
    kept = [
        g for g in groups
        if len(" ".join(s.text for s in g).split()) >= cfg.min_response_words
    ]
    dropped = [g for g in groups if g not in kept]

    mismatch = len(kept) != len(queries)

    pairs: list[QAPair] = []
    for k, q in enumerate(queries):
        pair = QAPair(
            pair_id=f"{script.clip_id}#q{k + 1}",
            query_utt_id=q.key,
            query_speaker=q.speaker,
            query_text=q.text,
            clip_id=script.clip_id,
            query_index=k,
            context=script.context_for(q, cfg.context_max_turns),
            count_mismatch=mismatch,
        )
        if include_reference:
            pair.reference_text = script.reference_for(q)

        if k < len(kept):
            grp = kept[k]
            text = " ".join(s.text.strip() for s in grp).strip()
            pair.response_text = text
            pair.response_start = grp[0].start
            pair.response_end = grp[-1].end
            pair.response_utt_ids = [s.utt_id for s in grp]
            pair.status = "answered" if text else "empty_asr"
        else:
            pair.status = "no_response"

        pairs.append(pair)

    leftover = [s for g in kept[len(queries):] for s in g]
    leftover += [s for g in dropped for s in g]
    return pairs, sorted(leftover, key=lambda u: u.start)


# --- time-based pairing for authored scripts --------------------------------

def build_pairs_by_time(
    script,                                  # authored.AuthoredScript
    query_times: dict[int, tuple[float, float]],   # query order -> (start, end)
    asr_segments: list[Utterance],
    cfg: PairingConfig,
    include_reference: bool = True,
) -> tuple[list[QAPair], list[Utterance]]:
    """Match agent bursts to questions on the clip's shared clock.

    ``query_times`` comes from aligning the authored turns to an ASR pass over
    ``input.wav``. A question missing from it has no known time; it is emitted
    as ``no_response`` rather than guessed at, so an alignment failure can
    never masquerade as a silent agent -- ``time_known`` records which is which.
    """
    queries = script.queries
    groups = _merge(sorted(asr_segments, key=lambda u: u.start), cfg.merge_gap_s)
    kept = [
        g for g in groups
        if len(" ".join(s.text for s in g).split()) >= cfg.min_response_words
    ]
    used: set[int] = set()

    starts = {k: query_times[k][0] for k in query_times}
    pairs: list[QAPair] = []

    for k, q in enumerate(queries):
        pair = QAPair(
            pair_id=f"{script.clip_id}#q{k + 1}",
            query_utt_id=q.key,
            query_speaker=q.speaker,
            query_text=q.text,
            clip_id=script.clip_id,
            query_index=k,
            context=script.context_for(q, cfg.context_max_turns),
        )
        if include_reference:
            pair.reference_text = script.reference_for(q)

        if k not in query_times:
            pair.status = "no_response"
            pair.time_known = False
            pairs.append(pair)
            continue

        q_start, q_end = query_times[k]
        pair.query_start, pair.query_end = q_start, q_end
        pair.time_known = True

        later = [starts[j] for j in starts if j > k]
        next_start = min(later) if later else float("inf")
        lo = q_end - cfg.overlap_tolerance_s
        hi = min(q_end + cfg.response_window_s, next_start + cfg.overlap_tolerance_s)

        chosen = None
        for gi, grp in enumerate(kept):
            if gi in used:
                continue
            if lo <= grp[0].start <= hi:
                chosen, used_gi = grp, gi
                used.add(gi)
                break

        if chosen:
            text = " ".join(s.text.strip() for s in chosen).strip()
            pair.response_text = text
            pair.response_start = chosen[0].start
            pair.response_end = chosen[-1].end
            pair.response_utt_ids = [s.utt_id for s in chosen]
            pair.status = "answered" if text else "empty_asr"
        else:
            pair.status = "no_response"

        pairs.append(pair)

    leftover = [s for gi, g in enumerate(kept) if gi not in used for s in g]
    leftover += [s for g in groups if g not in kept for s in g]
    return pairs, sorted(leftover, key=lambda u: u.start)
