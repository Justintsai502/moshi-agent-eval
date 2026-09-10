"""Batch evaluation over a clip pack.

Pack layout (phase3_p0_IF_evalset_subset):

    <pack>/ids.txt
    <pack>/clips/<clip_id>/input.wav      # user A+B mix
    <pack>/clips/<clip_id>/response.wav   # agent output, pad_sec longer
    <scripts>/<clip_id>_qa.json           # authored ground truth

Only `response.wav` is transcribed -- it is the agent track and the only thing
without a transcript. `input.wav` is never ASR'd: the human side is already
known verbatim from the authored script.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any

from . import asr as asr_mod
from .align import align_turns_to_asr
from .authored import AuthoredScript, load_authored_script
from .judge import judge_pairs
from .pairing import build_pairs_by_order, build_pairs_by_time
from .report import render_markdown, summarize
from .schema import QAPair, Utterance, Verdict, write_jsonl


@dataclass
class Clip:
    clip_id: str
    input_wav: str
    response_wav: str
    script_json: str


def discover_clips(pack_dir: str, scripts_dir: str) -> tuple[list[Clip], list[str]]:
    """Return (clips, problems). A clip is skipped if any piece is missing."""
    ids_path = os.path.join(pack_dir, "ids.txt")
    with open(ids_path, "r", encoding="utf-8") as fh:
        ids = [ln.strip() for ln in fh if ln.strip()]

    clips: list[Clip] = []
    problems: list[str] = []
    for cid in ids:
        inp = os.path.join(pack_dir, "clips", cid, "input.wav")
        resp = os.path.join(pack_dir, "clips", cid, "response.wav")
        scr = os.path.join(scripts_dir, f"{cid}_qa.json")
        missing = [p for p in (inp, resp, scr) if not os.path.exists(p)]
        if missing:
            problems.append(f"{cid}: missing {[os.path.basename(m) for m in missing]}")
            continue
        clips.append(Clip(cid, inp, resp, scr))
    return clips, problems


def clip_audio_stats(clip: Clip) -> dict[str, float]:
    di = asr_mod.audio_duration(clip.input_wav)
    dr = asr_mod.audio_duration(clip.response_wav)
    return {"input_s": di, "response_s": dr, "pad_s": dr - di}


# --- stages ------------------------------------------------------------------

def pack_asr(
    cfg, clips: list[Clip], out_dir: str
) -> tuple[dict[str, list[Utterance]], dict[str, list[Utterance]]]:
    """Transcribe the agent track for every clip, reusing one loaded model.

    In ``pairing.mode: time`` the user mix is transcribed too -- not for its
    text, which the authored script already gives verbatim, but purely to put
    the questions on the audio clock.
    """
    by_clip: dict[str, list[Utterance]] = {}
    input_by_clip: dict[str, list[Utterance]] = {}
    rows: list[dict[str, Any]] = []
    need_input = cfg.pairing.mode == "time"

    for i, clip in enumerate(clips, 1):
        script = load_authored_script(
            clip.script_json, clip.clip_id, cfg.data.agent_speaker
        )
        if cfg.asr.backend == "gold":
            # Oracle path: no model at all. Every authored turn is placed on
            # ONE shared fake clock by script index, then split into the two
            # tracks -- so response and input timestamps stay comparable, just
            # as the real stems do.
            slot = cfg.pairing.response_window_s * 0.8
            clock = {
                t.key: (i * slot, i * slot + 3.0)
                for i, t in enumerate(script.turns)
            }
            segs = [
                Utterance(
                    utt_id=f"{clip.clip_id}#{t.key}", speaker="AGENT",
                    start=clock[t.key][0], end=clock[t.key][1],
                    text=t.text, source="asr",
                )
                for t in script.agent_turns
            ]
        else:
            segs = asr_mod.transcribe(clip.response_wav, cfg.asr)

        by_clip[clip.clip_id] = segs
        rows += [{"clip_id": clip.clip_id, "track": "response", **asdict(s)} for s in segs]

        if need_input:
            if cfg.asr.backend == "gold":
                ins = [
                    Utterance(
                        utt_id=f"{clip.clip_id}#{t.key}", speaker="USER",
                        start=clock[t.key][0], end=clock[t.key][1],
                        text=t.text, source="asr",
                    )
                    for t in script.turns
                    if t.speaker != cfg.data.agent_speaker
                ]
            else:
                ins = asr_mod.transcribe(clip.input_wav, cfg.asr)
            input_by_clip[clip.clip_id] = ins
            rows += [
                {"clip_id": clip.clip_id, "track": "input", **asdict(s)} for s in ins
            ]

        print(
            f"  [asr {i}/{len(clips)}] {clip.clip_id}: "
            f"{len(segs)} response segment(s)"
            + (f", {len(input_by_clip[clip.clip_id])} input" if need_input else "")
        )

    write_jsonl(os.path.join(out_dir, "asr.jsonl"), rows)
    return by_clip, input_by_clip


def _question_times(
    script: AuthoredScript, input_segs: list[Utterance], cfg
) -> tuple[dict[int, tuple[float, float]], float]:
    """Place each question on the clip's clock via authored-to-ASR alignment.

    Only human *speech* is aligned. Backchannels are dropped first: ASR
    discards most of them anyway, so feeding them to the aligner just created
    unmatchable rows that dragged the reported coverage down without ever
    affecting a question.

    The coverage returned is over questions, not over all turns -- a clip
    whose two questions both landed is fully usable however many ordinary
    turns failed to match.
    """
    human = script.human_speech
    al = align_turns_to_asr(
        [t.text for t in human], [s.text for s in input_segs],
        min_score=cfg.pairing.align_min_score,
    )
    order_of = {t.key: k for k, t in enumerate(script.queries)}

    times: dict[int, tuple[float, float]] = {}
    for ti, (ai, _score) in al.matched.items():
        key = human[ti].key
        if key in order_of:
            seg = input_segs[ai]
            times[order_of[key]] = (seg.start, seg.end)

    q_cov = len(times) / len(order_of) if order_of else 1.0
    return times, q_cov


def pack_pair(
    cfg, clips: list[Clip], asr_by_clip: dict[str, list[Utterance]],
    input_by_clip: dict[str, list[Utterance]], out_dir: str
) -> tuple[list[QAPair], list[dict[str, Any]]]:
    all_pairs: list[QAPair] = []
    leftovers: list[dict[str, Any]] = []
    coverages: list[float] = []

    for clip in clips:
        script: AuthoredScript = load_authored_script(
            clip.script_json, clip.clip_id, cfg.data.agent_speaker
        )
        segs = asr_by_clip.get(clip.clip_id, [])

        if cfg.pairing.mode == "time":
            times, cov = _question_times(
                script, input_by_clip.get(clip.clip_id, []), cfg
            )
            coverages.append(cov)
            pairs, extra = build_pairs_by_time(
                script, times, segs, cfg.pairing,
                include_reference=cfg.judge.use_reference,
            )
        else:
            pairs, extra = build_pairs_by_order(
                script, segs, cfg.pairing,
                include_reference=cfg.judge.use_reference,
            )
        all_pairs += pairs
        leftovers += [{"clip_id": clip.clip_id, **asdict(s)} for s in extra]

    if coverages:
        import statistics
        full = sum(1 for c in coverages if c >= 1.0)
        print(
            f"[pack] question timing recovered: "
            f"{full}/{len(coverages)} clips fully, "
            f"median {statistics.median(coverages):.0%}, min {min(coverages):.0%}"
        )

    write_jsonl(os.path.join(out_dir, "pairs.jsonl"), all_pairs)
    write_jsonl(os.path.join(out_dir, "unmatched_agent_segments.jsonl"), leftovers)

    from .prompts import build_messages
    write_jsonl(
        os.path.join(out_dir, "prompts.jsonl"),
        [
            {
                "pair_id": p.pair_id,
                "clip_id": p.clip_id,
                "messages": build_messages(p, cfg.judge.use_reference),
            }
            for p in all_pairs if p.status == "answered"
        ],
    )
    return all_pairs, leftovers


def pack_report(
    cfg, pairs: list[QAPair], verdicts: list[Verdict],
    leftovers: list[dict[str, Any]], clips: list[Clip], out_dir: str,
) -> dict[str, Any]:
    summary = summarize(
        pairs, verdicts, unmatched=leftovers,
        judge_backend=cfg.judge.backend,
        meta={
            "pack": cfg.data.pack_dir,
            "scripts": cfg.data.scripts_dir,
            "n_clips": len(clips),
            "asr_backend": cfg.asr.backend,
            "asr_model": cfg.asr.model,
            "judge_backend": cfg.judge.backend,
            "judge_model": cfg.judge.model,
            "use_reference": cfg.judge.use_reference,
            "pairing_mode": cfg.pairing.mode,
        },
    )

    mismatched = sorted({p.clip_id for p in pairs if p.count_mismatch})
    summary["clips_with_count_mismatch"] = mismatched
    summary["clip_count_mismatch_rate"] = (
        len(mismatched) / len(clips) if clips else None
    )

    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    md = render_markdown(summary)
    if mismatched:
        md += (
            "\n## Clips where burst count != question count\n\n"
            "Order-based pairing may be off by one in these clips — check them:\n\n"
            + "\n".join(f"- `{c}`" for c in mismatched) + "\n"
        )
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    return summary


def run_pack(cfg) -> dict[str, Any]:
    out_dir = os.path.join(cfg.out_dir, cfg.run_name)
    os.makedirs(out_dir, exist_ok=True)

    clips, problems = discover_clips(cfg.data.pack_dir, cfg.data.scripts_dir)
    if problems:
        print(f"[warn] skipping {len(problems)} incomplete clip(s):")
        for p in problems[:10]:
            print(f"   {p}")
    if cfg.data.limit:
        clips = clips[: cfg.data.limit]
    print(f"[pack] {len(clips)} clip(s) to evaluate")

    asr_by_clip, input_by_clip = pack_asr(cfg, clips, out_dir)
    pairs, leftovers = pack_pair(cfg, clips, asr_by_clip, input_by_clip, out_dir)
    print(f"[pack] {len(pairs)} question(s) paired")

    verdicts = judge_pairs(pairs, cfg.judge)
    write_jsonl(os.path.join(out_dir, "verdicts.jsonl"), verdicts)

    summary = pack_report(cfg, pairs, verdicts, leftovers, clips, out_dir)
    with open(os.path.join(out_dir, "config.used.json"), "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, ensure_ascii=False, indent=2)
    return summary
