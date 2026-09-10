"""Offline tests for the clip-pack path. No model is ever loaded."""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from moshi_eval.align import align_turns_to_asr, token_f1  # noqa: E402
from moshi_eval.authored import load_authored_script  # noqa: E402
from moshi_eval.config import load_config  # noqa: E402
from moshi_eval.pack import discover_clips, run_pack  # noqa: E402
from moshi_eval.pairing import PairingConfig, build_pairs_by_order, build_pairs_by_time  # noqa: E402
from moshi_eval.schema import Utterance  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PACK = os.path.join(ROOT, "bathrooms/kcire/mpmoshi-ifami/packs/phase3_p0_IF_evalset_subset")
SCRIPTS = os.path.join(ROOT, "authored_scripts")
CLIP = "ai_agent_3d_printing"


class Skip(Exception):
    """Raised when the data a test needs is not present in this checkout."""


def need(path: str) -> str:
    """Data-dependent tests skip rather than fail on a checkout without data."""
    if not os.path.exists(path):
        raise Skip(f"missing {os.path.relpath(path, ROOT)}")
    return path



def test_pack_discovery_is_complete():
    need(PACK); need(SCRIPTS)
    clips, problems = discover_clips(PACK, SCRIPTS)
    assert len(clips) == 49, f"expected 49 clips, got {len(clips)}"
    assert problems == [], problems
    for c in clips:
        assert os.path.exists(c.input_wav) and os.path.exists(c.response_wav)


def test_authored_script_structure():
    need(PACK); need(SCRIPTS)
    s = load_authored_script(os.path.join(SCRIPTS, f"{CLIP}_qa.json"), CLIP)
    assert [t.key for t in s.queries] == ["A,4", "A,11"]
    assert len(s.agent_turns) == 2
    # each question maps to the agent turn that follows it, not any other
    assert s.reference_for(s.queries[0]).startswith("PLA plastic")
    assert "0.1 to 0.3" in s.reference_for(s.queries[1])
    # the agent's scripted answer must not leak into the judge's context
    ctx = "\n".join(s.context_for(s.queries[1]))
    assert "PLA plastic" not in ctx


def test_every_clip_has_two_questions_and_two_references():
    need(PACK); need(SCRIPTS)
    clips, _ = discover_clips(PACK, SCRIPTS)
    for c in clips:
        s = load_authored_script(c.script_json, c.clip_id)
        assert len(s.queries) == 2, f"{c.clip_id}: {len(s.queries)} queries"
        assert all(s.reference_for(q) for q in s.queries), c.clip_id


def test_alignment_survives_realistic_asr_noise():
    need(PACK); need(SCRIPTS)
    """Dropped backchannels, a split turn and word errors must not break it."""
    s = load_authored_script(os.path.join(SCRIPTS, f"{CLIP}_qa.json"), CLIP)
    human = [t for t in s.turns if t.speaker != "C"]

    asr = []
    for t in human:
        if t.function == "backchannel":
            continue                                    # ASR often drops these
        txt = t.text.lower().replace(",", "").replace("—", "")
        if "material is most commonly" in txt:
            txt = txt.replace("beginner", "beginer")    # word error
        if "how thick is" in txt:                       # split into two segments
            half = len(txt) // 2
            asr += [txt[:half], txt[half:]]
            continue
        asr.append(txt)

    al = align_turns_to_asr([t.text for t in human], asr)
    q_idx = [i for i, t in enumerate(human) if "C" in t.addressing]
    assert all(i in al.matched for i in q_idx), "a question failed to align"
    # and it must map to the right segment, not merely to something
    first = asr[al.matched[q_idx[0]][0]]
    assert "material" in first


def test_token_f1_discriminates():
    assert token_f1("how thick is a layer", "how thick is a layer") == 1.0
    assert token_f1("how thick is a layer", "what material for printing") < 0.2


def test_order_pairing_flags_burst_count_mismatch():
    need(PACK); need(SCRIPTS)
    s = load_authored_script(os.path.join(SCRIPTS, f"{CLIP}_qa.json"), CLIP)
    # three bursts against two questions -> every pair flagged
    segs = [
        Utterance(f"a{i}", "AGENT", i * 20.0, i * 20.0 + 3.0, f"answer number {i}", source="asr")
        for i in range(3)
    ]
    pairs, extra = build_pairs_by_order(s, segs, PairingConfig(), include_reference=True)
    assert len(pairs) == 2
    assert all(p.count_mismatch for p in pairs)
    assert len(extra) == 1


def test_order_pairing_drops_short_blips():
    need(PACK); need(SCRIPTS)
    s = load_authored_script(os.path.join(SCRIPTS, f"{CLIP}_qa.json"), CLIP)
    segs = [
        Utterance("blip", "AGENT", 3.7, 4.2, "uh", source="asr"),
        Utterance("a1", "AGENT", 20.0, 23.0, "PLA plastic is common", source="asr"),
        Utterance("a2", "AGENT", 50.0, 53.0, "about zero point two millimeters", source="asr"),
    ]
    pairs, _ = build_pairs_by_order(s, segs, PairingConfig(min_response_words=2))
    assert not any(p.count_mismatch for p in pairs)   # blip dropped -> 2 vs 2
    assert pairs[0].response_text.startswith("PLA")


def test_time_pairing_reports_unknown_time_distinctly():
    need(PACK); need(SCRIPTS)
    """An unalignable question must not be reported as a silent agent."""
    s = load_authored_script(os.path.join(SCRIPTS, f"{CLIP}_qa.json"), CLIP)
    segs = [Utterance("a1", "AGENT", 20.0, 23.0, "PLA plastic is common", source="asr")]
    pairs, _ = build_pairs_by_time(s, {0: (10.0, 18.0)}, segs, PairingConfig())
    assert pairs[0].status == "answered" and pairs[0].time_known
    assert pairs[1].status == "no_response" and not pairs[1].time_known


def test_full_pack_offline_run():
    need(PACK); need(SCRIPTS)
    with tempfile.TemporaryDirectory() as d:
        cfg = load_config(os.path.join(ROOT, "config/pack_offline.yaml"),
                          overrides=[f'out_dir="{d}"', "data.limit=5"])
        cfg.data.pack_dir = PACK
        cfg.data.scripts_dir = SCRIPTS
        summary = run_pack(cfg)
        assert summary["mock_run"] is True
        assert summary["accuracy_strict"] is None      # never score a mock run
        assert summary["total_queries"] == 10          # 5 clips x 2 questions
        run = os.path.join(d, cfg.run_name)
        for f in ("asr.jsonl", "pairs.jsonl", "prompts.jsonl", "verdicts.jsonl",
                  "report.md", "report.json"):
            assert os.path.exists(os.path.join(run, f)), f
        prompts = [json.loads(l) for l in open(os.path.join(run, "prompts.jsonl"))]
        assert len(prompts) == 10
        assert all(m["messages"][0]["role"] == "system" for m in prompts)


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
