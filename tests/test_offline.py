"""Offline tests. No model is ever loaded; heavy backends must stay blocked."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from moshi_eval.asr import ASRConfig, transcribe  # noqa: E402
from moshi_eval.config import load_config  # noqa: E402
from moshi_eval.ground_truth import (  # noqa: E402
    find_overlaps, load_ground_truth, natural_key, sort_utterances,
)
from moshi_eval.judge import HeavyModelBlocked, JudgeConfig, build_backend, parse_verdict  # noqa: E402
from moshi_eval.pairing import PairingConfig, build_pairs  # noqa: E402
from moshi_eval.pipeline import run_all  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GT = os.path.join(ROOT, "data/sample_ai_agent_road_trip/aligned_script.jsonl")


def test_ground_truth_sorted_and_split():
    gt = load_ground_truth(GT, agent_speaker="C")
    assert [u.utt_id for u in gt.agent_turns] == ["g002", "g004", "g007"]
    # g006 precedes g005 in the file but starts later -> must be reordered
    ids = [u.utt_id for u in gt.human_turns]
    assert ids.index("g006") < ids.index("g005")
    # backchannel g005 is not a query
    assert [u.utt_id for u in gt.queries] == ["g001", "g003", "g006"]


def test_natural_id_order_beats_string_order():
    assert natural_key("g9") < natural_key("g10")      # string order says otherwise
    assert natural_key("g001") < natural_key("g002")
    assert sorted(["g10", "g9", "g100"], key=natural_key) == ["g9", "g10", "g100"]


def test_simultaneous_start_resolved_by_role_not_jitter():
    """g006/g005 start 11.7 ms apart; order must not hinge on that."""
    gt = load_ground_truth(GT, agent_speaker="C")
    rows = sort_utterances(gt.human_turns + gt.agent_turns)
    ids = [u.utt_id for u in rows]
    assert ids.index("g006") < ids.index("g005")

    # flip the jitter: make the backchannel start marginally FIRST
    flipped = gt.human_turns + gt.agent_turns
    bc = next(u for u in flipped if u.utt_id == "g005")
    sp = next(u for u in flipped if u.utt_id == "g006")
    bc.start, sp.start = sp.start - 0.002, bc.start + 0.002
    ids2 = [u.utt_id for u in sort_utterances(flipped)]
    assert ids2.index("g006") < ids2.index("g005"), (
        "speech must still precede the backchannel layered onto it"
    )


def test_large_gaps_still_ordered_strictly_by_time():
    """The tie-break must apply only within the simultaneity window."""
    gt = load_ground_truth(GT, agent_speaker="C")
    rows = sort_utterances(gt.human_turns + gt.agent_turns)
    far = [u for u in rows if u.utt_id not in ("g005", "g006")]
    assert [u.utt_id for u in far] == [
        u.utt_id for u in sorted(far, key=lambda u: u.start)
    ]


def test_overlap_detection():
    gt = load_ground_truth(GT, agent_speaker="C")
    ov = find_overlaps(gt.human_turns + gt.agent_turns)
    assert len(ov) == 1
    a, b, secs = ov[0]
    assert {a, b} == {"g005", "g006"}
    assert abs(secs - 6.24) < 0.01          # min(ends) - later start


def test_pairing_matches_each_query():
    gt = load_ground_truth(GT, agent_speaker="C")
    segs = transcribe("", ASRConfig(backend="gold"), gt.agent_turns)
    pairs = build_pairs(gt, segs, PairingConfig())
    assert len(pairs) == 3
    assert all(p.status == "answered" for p in pairs)
    assert "128" in pairs[1].response_text
    assert all(p.latency_ms is not None and p.latency_ms >= 0 for p in pairs)


def test_reference_does_not_leak_across_queries():
    """p001's reference must be g002 only -- not g004, the next answer."""
    gt = load_ground_truth(GT, agent_speaker="C")
    segs = transcribe("", ASRConfig(backend="gold"), gt.agent_turns)
    pairs = build_pairs(gt, segs, PairingConfig(), include_reference=True)
    refs = [p.reference_text for p in pairs]
    assert refs[0] is not None and "128" not in refs[0]
    assert refs[1] is not None and "128" in refs[1]
    assert len(set(refs)) == len(refs)          # no reference reused


def test_heavy_backends_blocked_without_env():
    os.environ.pop("MOSHI_EVAL_ALLOW_HEAVY", None)
    for backend in ("vllm", "transformers"):
        try:
            build_backend(JudgeConfig(backend=backend))
        except HeavyModelBlocked:
            continue
        except ImportError:
            raise AssertionError(f"{backend} attempted an import before the guard")
        raise AssertionError(f"{backend} was not blocked")


def test_verdict_parsing():
    cfg = JudgeConfig()
    raw = '<think>hmm</think>\n```json\n{"verdict":"correct","confidence":0.9,' \
          '"answer_summary":"30-40 mpg","reason":"matches"}\n```'
    v = parse_verdict(raw, "p001", cfg)
    assert v.parse_ok and v.verdict == "correct" and v.confidence == 0.9

    bad = parse_verdict("I think it is fine", "p002", cfg)
    assert not bad.parse_ok and bad.verdict == "unintelligible"


def test_full_offline_pipeline(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cfg = load_config(
            os.path.join(ROOT, "config/sample_offline.yaml"),
            overrides=[f'out_dir="{d}"'],
        )
        cfg.data.ground_truth = GT
        cfg.data.agent_audio = os.path.join(
            ROOT, "data/sample_ai_agent_road_trip/audio_qwen3-tts-1.7b-custom_C.wav"
        )
        summary = run_all(cfg)
        assert summary["mock_run"] is True
        assert summary["accuracy_strict"] is None      # mock must never score
        assert summary["total_queries"] == 3
        assert os.path.exists(os.path.join(d, cfg.run_name, "report.md"))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")
