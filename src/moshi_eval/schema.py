"""Core data structures shared by every stage of the pipeline.

Deliberately dependency-free so the whole schema can be imported on a laptop
without pulling in torch / vllm / whisper.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Iterator


@dataclass
class Utterance:
    """One spoken turn, from ground truth or from ASR."""

    utt_id: str
    speaker: str
    start: float
    end: float
    text: str
    function: str = "speech"          # speech | backchannel | ...
    subtype: str | None = None
    addressing: list[str] = field(default_factory=list)
    source: str = "gold"              # gold | asr
    asr_confidence: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    @classmethod
    def from_gold(cls, row: dict[str, Any]) -> "Utterance":
        return cls(
            utt_id=row["utt_id"],
            speaker=row["speaker"],
            start=float(row["start"]),
            end=float(row["end"]),
            text=row.get("text", ""),
            function=row.get("function", "speech"),
            subtype=row.get("subtype"),
            addressing=list(row.get("addressing") or []),
            source="gold",
        )


@dataclass
class QAPair:
    """A human query matched against whatever the agent track said next."""

    pair_id: str
    query_utt_id: str
    query_speaker: str
    query_text: str
    # Timestamps are None for authored scripts, which carry order but no clock.
    query_start: float | None = None
    query_end: float | None = None
    clip_id: str = ""
    query_index: int | None = None      # 0-based order within the clip

    response_text: str = ""
    response_start: float | None = None
    response_end: float | None = None
    response_utt_ids: list[str] = field(default_factory=list)

    # Optional gold answer. Present when calibrating against the scripted TTS
    # run; absent for real Moshi output, which is what this pipeline is for.
    reference_text: str | None = None

    context: list[str] = field(default_factory=list)   # rendered dialogue context
    status: str = "answered"                          # answered | no_response | empty_asr
    # True when the clip produced a different number of agent bursts than it
    # had questions, so this order-based match may be off by one.
    count_mismatch: bool = False
    # False when the question could not be placed on the audio clock, so
    # "no_response" means "alignment failed", not "the agent stayed silent".
    time_known: bool = True

    @property
    def latency_ms(self) -> float | None:
        if self.response_start is None or self.query_end is None:
            return None
        return (self.response_start - self.query_end) * 1000.0


@dataclass
class Verdict:
    """One judge decision for one QAPair."""

    pair_id: str
    verdict: str            # see VERDICTS
    confidence: float | None = None
    reason: str = ""
    answer_summary: str = ""
    judge_model: str = ""
    backend: str = ""
    is_mock: bool = False
    raw_output: str = ""
    parse_ok: bool = True


VERDICTS = (
    "correct",             # factually right and responsive
    "partially_correct",   # right direction, incomplete or imprecise
    "incorrect",           # factually wrong
    "not_responsive",      # said something, but did not answer the question
    "no_answer",           # agent track silent after the query
    "unintelligible",      # ASR output too degraded to judge
    "mock_unjudged",       # produced by the mock backend, NOT a real result
)

PAIR_STATUS = (
    "answered",
    "no_response",       # agent track silent in the matched slot
    "empty_asr",
    "count_mismatch",    # answered, but the clip's burst count != question count
)


# --- jsonl helpers -----------------------------------------------------------

def read_jsonl(path: str) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[Any]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            obj = row if isinstance(row, dict) else asdict(row)
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n
