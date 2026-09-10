"""Offline tests for the timed-ground-truth path. No model is ever loaded."""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from moshi_eval.config import load_config  # noqa: E402
from moshi_eval.pack import run_timed  # noqa: E402
from moshi_eval.pairing import is_noise  # noqa: E402
from moshi_eval.schema import Utterance  # noqa: E402
from moshi_eval.timed import (  # noqa: E402
    detect_agent_speaker, discover_timed_clips, load_timed_script,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GT = os.path.join(ROOT, "authored_scripts/synthesized_if_dataset_4spk")
EVAL = os.path.join(ROOT, "phase3_p0_IF4_evalset")
IDS = os.path.join(EVAL, "ids.txt")


class Skip(Exception):
    """Raised when the data a test needs is not present in this checkout."""


def need(path: str) -> str:
    """Data-dependent tests skip rather than fail on a checkout without data."""
    if not os.path.exists(path):
        raise Skip(f"missing {os.path.relpath(path, ROOT)}")
    return path



def test_agent_speaker_is_detected_not_assumed():
    need(GT); need(IDS)
    """It was C in the 3-speaker packs and is D here."""
    s = load_timed_script(
        os.path.join(GT, "ai_agent4_3d_printer_troubleshooting/aligned_script.jsonl"),
        "ai_agent4_3d_printer_troubleshooting",
    )
    assert s.agent_speaker == "D"
    assert all(u.speaker == "D" for u in s.agent_turns)


def test_continuer_turns_count_as_questions():
    need(GT); need(IDS)
    """443 questions carry function='continuer'; requiring 'speech' lost them."""
    s = load_timed_script(
        os.path.join(GT, "ai_agent4_assembling_a_kayak_rack/aligned_script.jsonl"),
        "ai_agent4_assembling_a_kayak_rack",
    )
    fns = {q.function for q in s.queries}
    assert len(s.queries) == 2, f"expected 2 questions, got {len(s.queries)}"
    assert "continuer" in fns, "the continuer-tagged question was dropped"


def test_exploded_addressing_is_repaired():
    u = Utterance.from_gold({
        "utt_id": "x", "speaker": "C", "start": 0.0, "end": 1.0, "text": "hi",
        "addressing": ["g", "r", "o", "u", "p"],
    })
    assert u.addressing == ["group"]
    # a genuine multi-target list must survive untouched
    v = Utterance.from_gold({
        "utt_id": "y", "speaker": "C", "start": 0.0, "end": 1.0, "text": "hi",
        "addressing": ["A", "B"],
    })
    assert v.addressing == ["A", "B"]


def test_one_word_answers_are_not_noise():
    """The agent answers factual questions with a single word."""
    assert not is_noise("Seven.", 1)
    assert not is_noise("7.", 1)
    assert not is_noise("128 ounces.", 1)
    assert is_noise("", 1)
    assert is_noise("uh", 1)
    assert is_noise("um uh hmm", 1)


def test_question_times_come_from_the_script():
    need(GT); need(IDS)
    s = load_timed_script(
        os.path.join(GT, "ai_agent4_3d_printer_troubleshooting/aligned_script.jsonl"),
        "ai_agent4_3d_printer_troubleshooting",
    )
    times = s.query_times()
    assert set(times) == {0, 1}
    for k, q in enumerate(s.queries):
        assert times[k] == (q.start, q.end)
    # reference must be the answer to THIS question, not the next one
    assert "60 degrees Celsius" in s.reference_for(s.queries[0])
    assert "140 degrees Fahrenheit" in s.reference_for(s.queries[1])


def test_discovery_covers_the_whole_eval_pack():
    need(GT); need(IDS)
    clips, problems = discover_timed_clips(
        scripts_root=GT,
        agent_audio_dir=os.path.join(EVAL, "clips"),
        agent_audio_name="response.wav",
        ids_file=IDS,
    )
    assert len(clips) == 155, f"{len(clips)} clips, {problems[:3]}"
    assert problems == []
    for c in clips[:5]:
        assert c.agent_wav.endswith("response.wav") and os.path.exists(c.agent_wav)


def test_gold_oracle_matches_every_question():
    need(GT); need(IDS)
    """With scripted text as ASR, nothing may go unmatched or unanswered."""
    with tempfile.TemporaryDirectory() as d:
        cfg = load_config(os.path.join(ROOT, "config/if4_offline.yaml"),
                          overrides=[f'out_dir="{d}"'])
        cfg.data.scripts_root = GT
        cfg.data.ids_file = IDS
        cfg.data.agent_audio_dir = os.path.join(EVAL, "clips")
        summary = run_timed(cfg)

        assert summary["mock_run"] is True
        assert summary["accuracy_strict"] is None
        assert summary["total_queries"] == 310, summary["total_queries"]
        assert summary["no_answer_rate"] == 0.0
        assert summary["unmatched_agent_segments"] == 0
        assert summary["meta"]["agent_speaker"] == ["D"]

        pairs = [
            json.loads(l)
            for l in open(os.path.join(d, cfg.run_name, "pairs.jsonl"))
        ]
        assert all(p["time_known"] for p in pairs)
        assert all(p["reference_text"] for p in pairs)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ran = skipped = 0
    for fn in fns:
        try:
            fn()
        except Skip as exc:
            print(f"SKIP {fn.__name__}  ({exc})")
            skipped += 1
            continue
        print(f"PASS {fn.__name__}")
        ran += 1
    print(f"\n{ran} passed, {skipped} skipped")
