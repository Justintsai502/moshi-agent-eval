"""Clips that ship their own timed ground truth.

Layout (synthesized_if_dataset_4spk):

    <root>/<clip_id>/aligned_script.jsonl          start/end per turn
    <root>/<clip_id>/audio_<model>.wav             full mix
    <root>/<clip_id>/audio_<model>_<SPK>.wav       per-speaker stems
    <root>/<clip_id>/tts_info.json                 voice map

Because the script already carries timestamps, this path needs no ASR of the
user mix and no authored-to-ASR alignment: question times are read straight
off the file. Only the agent's audio is transcribed.

The agent speaker is NOT fixed. It was C in the earlier 3-speaker packs and is
D here, so it is detected from the data rather than assumed.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .ground_truth import sort_utterances
from .schema import Utterance, read_jsonl

WAKE_RE = re.compile(r"\bai\s*agent\b", re.IGNORECASE)


def is_speech(u: Utterance) -> bool:
    """Everything that is not a backchannel counts as speaking.

    The label is not always "speech": 541 turns are tagged "continuer" (a
    speaker carrying on from their own previous turn), and 443 of those are
    questions to the agent -- 15% of the whole question set. Testing for
    `!= "backchannel"` rather than `== "speech"` also keeps any future label
    from silently dropping questions.
    """
    return u.function != "backchannel"


def detect_agent_speaker(turns: list[Utterance], fallback: str = "D") -> str:
    """Whoever the wake-word questions are addressed to.

    Counting `addressing` on turns that say "AI Agent" is robust to the label
    moving between datasets; it also never mistakes a human who happens to
    answer a lot for the agent.
    """
    votes: dict[str, int] = {}
    for u in turns:
        if is_speech(u) and WAKE_RE.search(u.text):
            for a in u.addressing:
                votes[a] = votes.get(a, 0) + 1
    if not votes:
        return fallback
    return max(votes.items(), key=lambda kv: kv[1])[0]


@dataclass
class TimedScript:
    """Same interface the pairing code expects, backed by timestamped turns."""

    clip_id: str
    turns: list[Utterance]
    agent_speaker: str

    @property
    def queries(self) -> list[Utterance]:
        return [
            u for u in self.turns
            if u.speaker != self.agent_speaker
            and is_speech(u)
            and self.agent_speaker in u.addressing
            and u.text.strip()
        ]

    @property
    def agent_turns(self) -> list[Utterance]:
        return [u for u in self.turns if u.speaker == self.agent_speaker]

    @property
    def human_speech(self) -> list[Utterance]:
        return [
            u for u in self.turns
            if u.speaker != self.agent_speaker and is_speech(u)
        ]

    def query_times(self) -> dict[int, tuple[float, float]]:
        """Question order -> (start, end), read directly off the script."""
        return {k: (q.start, q.end) for k, q in enumerate(self.queries)}

    def reference_for(self, query: Utterance) -> str | None:
        """The scripted agent turn answering this question.

        Bounded by the next question so a later answer cannot leak in.
        """
        later_q = [q.start for q in self.queries if q.start > query.start]
        limit = min(later_q) if later_q else float("inf")
        hits = [
            u.text for u in self.agent_turns
            if query.end - 1e-6 <= u.start < limit
        ]
        return " ".join(hits).strip() or None

    def context_for(self, query: Utterance, max_turns: int = 8) -> list[str]:
        """Preceding human speech. Backchannels and agent lines are omitted."""
        prior = [u for u in self.human_speech if u.end <= query.start + 1e-6]
        return [f"Speaker {u.speaker}: {u.text}" for u in prior[-max_turns:]]


def load_timed_script(
    path: str, clip_id: str = "", agent_speaker: str | None = None
) -> TimedScript:
    turns = sort_utterances([Utterance.from_gold(r) for r in read_jsonl(path)])
    spk = agent_speaker or detect_agent_speaker(turns)
    return TimedScript(clip_id=clip_id or path, turns=turns, agent_speaker=spk)


# --- clip discovery ----------------------------------------------------------

@dataclass
class TimedClip:
    clip_id: str
    script: str
    agent_wav: str
    mix_wav: str | None = None


def _find_audio(clip_dir: str, agent_speaker: str) -> tuple[str | None, str | None]:
    """Locate the agent stem and the full mix inside one clip directory."""
    wavs = sorted(f for f in os.listdir(clip_dir) if f.endswith(".wav"))
    stem_suffix = f"_{agent_speaker}.wav"
    agent = next((f for f in wavs if f.endswith(stem_suffix)), None)
    mix = next(
        (f for f in wavs if not re.search(r"_[A-Z]\.wav$", f)), None
    )
    return (
        os.path.join(clip_dir, agent) if agent else None,
        os.path.join(clip_dir, mix) if mix else None,
    )


def discover_timed_clips(
    scripts_root: str,
    agent_audio_dir: str = "",
    agent_audio_name: str = "response.wav",
    ids_file: str = "",
    agent_speaker: str | None = None,
    limit: int = 0,
) -> tuple[list[TimedClip], list[str]]:
    """Enumerate clips, pairing timed ground truth with the agent's audio.

    ``agent_audio_dir`` points at the model's outputs, laid out as
    ``<dir>/<clip_id>/<agent_audio_name>``. Leave it empty to fall back to the
    scripted agent's own TTS stem inside the ground-truth directory, which
    evaluates the reference system against itself -- the calibration run.

    ``ids_file`` restricts the set to the ids an eval pack actually shipped;
    without it every directory under ``scripts_root`` is taken.
    """
    if ids_file:
        with open(ids_file, encoding="utf-8") as fh:
            ids = [ln.strip() for ln in fh if ln.strip()]
    else:
        ids = sorted(
            d for d in os.listdir(scripts_root)
            if os.path.isdir(os.path.join(scripts_root, d))
        )

    clips: list[TimedClip] = []
    problems: list[str] = []

    for cid in ids:
        clip_dir = os.path.join(scripts_root, cid)
        script = os.path.join(clip_dir, "aligned_script.jsonl")
        if not os.path.exists(script):
            problems.append(f"{cid}: no aligned_script.jsonl")
            continue

        spk = agent_speaker or detect_agent_speaker(
            [Utterance.from_gold(r) for r in read_jsonl(script)]
        )
        _, mix = _find_audio(clip_dir, spk)

        if agent_audio_dir:
            agent = os.path.join(agent_audio_dir, cid, agent_audio_name)
        else:
            agent, _ = _find_audio(clip_dir, spk)

        if not agent or not os.path.exists(agent):
            problems.append(f"{cid}: agent audio missing ({agent or f'stem {spk}'})")
            continue

        clips.append(TimedClip(cid, script, agent, mix))
        if limit and len(clips) >= limit:
            break

    return clips, problems
