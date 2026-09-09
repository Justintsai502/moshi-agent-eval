"""Stage functions. Each writes its artefact so stages can run on different
machines: ASR + judge on the server, pairing and reporting anywhere.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any

from . import asr as asr_mod
from .config import Config
from .ground_truth import load_ground_truth
from .judge import judge_pairs
from .pairing import build_pairs, unmatched_segments
from .prompts import build_messages
from .report import render_markdown, summarize
from .schema import QAPair, Utterance, Verdict, read_jsonl, write_jsonl


def run_dir(cfg: Config) -> str:
    d = os.path.join(cfg.out_dir, cfg.run_name)
    os.makedirs(d, exist_ok=True)
    return d


def _p(cfg: Config, name: str) -> str:
    return os.path.join(run_dir(cfg), name)


# --- stages ------------------------------------------------------------------

def stage_asr(cfg: Config) -> list[Utterance]:
    gt = load_ground_truth(
        cfg.data.ground_truth, cfg.data.agent_speaker, cfg.data.query_selector
    )
    segs = asr_mod.transcribe(cfg.data.agent_audio, cfg.asr, gt.agent_turns)
    write_jsonl(_p(cfg, "asr.jsonl"), segs)
    return segs


def stage_pair(cfg: Config, segs: list[Utterance] | None = None) -> list[QAPair]:
    gt = load_ground_truth(
        cfg.data.ground_truth, cfg.data.agent_speaker, cfg.data.query_selector
    )
    if segs is None:
        segs = [Utterance(**r) for r in read_jsonl(_p(cfg, "asr.jsonl"))]

    pairs = build_pairs(gt, segs, cfg.pairing, include_reference=cfg.judge.use_reference)
    write_jsonl(_p(cfg, "pairs.jsonl"), pairs)
    write_jsonl(
        _p(cfg, "unmatched_agent_segments.jsonl"), unmatched_segments(segs, pairs)
    )
    # Prompts are dumped whether or not a judge runs -- this is the artefact you
    # eyeball on the laptop before shipping the job to the server.
    write_jsonl(
        _p(cfg, "prompts.jsonl"),
        [
            {"pair_id": p.pair_id, "messages": build_messages(p, cfg.judge.use_reference)}
            for p in pairs
            if p.status == "answered"
        ],
    )
    return pairs


def stage_judge(cfg: Config, pairs: list[QAPair] | None = None) -> list[Verdict]:
    if pairs is None:
        pairs = [QAPair(**r) for r in read_jsonl(_p(cfg, "pairs.jsonl"))]
    verdicts = judge_pairs(pairs, cfg.judge)
    write_jsonl(_p(cfg, "verdicts.jsonl"), verdicts)
    return verdicts


def stage_report(
    cfg: Config,
    pairs: list[QAPair] | None = None,
    verdicts: list[Verdict] | None = None,
) -> dict[str, Any]:
    if pairs is None:
        pairs = [QAPair(**r) for r in read_jsonl(_p(cfg, "pairs.jsonl"))]
    if verdicts is None:
        verdicts = [Verdict(**r) for r in read_jsonl(_p(cfg, "verdicts.jsonl"))]

    unmatched_path = _p(cfg, "unmatched_agent_segments.jsonl")
    unmatched = (
        [Utterance(**r) for r in read_jsonl(unmatched_path)]
        if os.path.exists(unmatched_path) else []
    )

    summary = summarize(
        pairs,
        verdicts,
        unmatched,
        judge_backend=cfg.judge.backend,
        meta={
            "run_name": cfg.run_name,
            "ground_truth": cfg.data.ground_truth,
            "agent_audio": cfg.data.agent_audio,
            "asr_backend": cfg.asr.backend,
            "asr_model": cfg.asr.model,
            "judge_backend": cfg.judge.backend,
            "judge_model": cfg.judge.model,
            "use_reference": cfg.judge.use_reference,
        },
    )
    with open(_p(cfg, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    with open(_p(cfg, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(render_markdown(summary))
    with open(_p(cfg, "config.used.json"), "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, ensure_ascii=False, indent=2)
    return summary


def run_all(cfg: Config) -> dict[str, Any]:
    segs = stage_asr(cfg)
    pairs = stage_pair(cfg, segs)
    verdicts = stage_judge(cfg, pairs)
    return stage_report(cfg, pairs, verdicts)
