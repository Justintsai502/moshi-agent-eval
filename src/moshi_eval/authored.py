"""Loader for `authored_scripts/<clip_id>_qa.json`.

Shape: a dict keyed "<speaker>,<index>" -- ordering only, NO timestamps.

    {"A,0": {"action":"speak","function":"speech","addressing":["B"],
             "text":"The bed needs to be perfectly level..."},
     "A,4": {... "addressing":["C"], "text":"AI Agent, what material ..."},
     "C,5": {... "addressing":["A"], "text":"PLA plastic is ..."}}

Speaker C is the AI agent. Its turns are the *intended* answers, so they serve
as reference answers for judging what Moshi actually produced.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

KEY_RE = re.compile(r"^([A-Za-z]+)\s*,\s*(\d+)$")


@dataclass
class AuthoredTurn:
    key: str
    speaker: str
    index: int
    text: str
    function: str = "speech"
    subtype: str | None = None
    addressing: list[str] = field(default_factory=list)


@dataclass
class AuthoredScript:
    clip_id: str
    turns: list[AuthoredTurn]
    agent_speaker: str = "C"

    @property
    def queries(self) -> list[AuthoredTurn]:
        """Human turns addressed to the agent, in script order."""
        return [
            t for t in self.turns
            if t.speaker != self.agent_speaker
            and t.function == "speech"
            and self.agent_speaker in t.addressing
            and t.text.strip()
        ]

    @property
    def agent_turns(self) -> list[AuthoredTurn]:
        return [t for t in self.turns if t.speaker == self.agent_speaker]

    @property
    def human_speech(self) -> list[AuthoredTurn]:
        """Human turns that are actual speech.

        Backchannels are excluded from the whole pipeline. Measured over 400
        timed scripts: every one of the 958 overlapping turn pairs involves a
        backchannel, and speech never overlaps speech. Dropping them leaves a
        strictly sequential timeline -- and ASR drops most of them anyway, so
        keeping them only poisoned the authored-to-ASR alignment.
        """
        return [
            t for t in self.turns
            if t.speaker != self.agent_speaker
            and t.function == "speech"
            and t.text.strip()
        ]

    def reference_for(self, query: AuthoredTurn) -> str | None:
        """The agent turn that directly follows this query in the script."""
        after = [t for t in self.agent_turns if t.index > query.index]
        return after[0].text.strip() if after else None

    def context_for(self, query: AuthoredTurn, max_turns: int = 8) -> list[str]:
        """Preceding human speech, verbatim from the script (no ASR error).

        Backchannels are omitted, and so are the agent's own scripted lines --
        the judge must never see the intended answer as conversation history.
        """
        prior = [t for t in self.human_speech if t.index < query.index]
        return [f"Speaker {t.speaker}: {t.text}" for t in prior[-max_turns:]]


def load_authored_script(path: str, clip_id: str = "", agent_speaker: str = "C") -> AuthoredScript:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    turns: list[AuthoredTurn] = []
    for key, val in raw.items():
        m = KEY_RE.match(key)
        if not m:
            raise ValueError(f"{path}: unparseable turn key {key!r}")
        speaker, idx = m.group(1), int(m.group(2))
        turns.append(
            AuthoredTurn(
                key=key,
                speaker=speaker,
                index=idx,
                text=str(val.get("text", "")),
                function=val.get("function", "speech"),
                subtype=val.get("subtype"),
                addressing=list(val.get("addressing") or []),
            )
        )

    turns.sort(key=lambda t: t.index)
    return AuthoredScript(
        clip_id=clip_id or path, turns=turns, agent_speaker=agent_speaker
    )
