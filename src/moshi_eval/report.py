"""Aggregate verdicts into a JSON summary plus a readable markdown report."""

from __future__ import annotations

import statistics
from typing import Any

from .schema import QAPair, Utterance, Verdict

SCORED_CORRECT = {"correct"}
SCORED_PARTIAL = {"partially_correct"}


def summarize(
    pairs: list[QAPair],
    verdicts: list[Verdict],
    unmatched: list[Utterance] | None = None,
    meta: dict[str, Any] | None = None,
    judge_backend: str | None = None,
) -> dict[str, Any]:
    by_id = {v.pair_id: v for v in verdicts}
    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v.verdict] = counts.get(v.verdict, 0) + 1

    total = len(pairs)
    # A run configured for the mock judge counts as mock even if every pair
    # happened to be settled by rule, so no scores are ever reported from one.
    is_mock = any(v.is_mock for v in verdicts) or judge_backend == "mock"
    n_correct = sum(counts.get(k, 0) for k in SCORED_CORRECT)
    n_partial = sum(counts.get(k, 0) for k in SCORED_PARTIAL)

    latencies = [p.latency_ms for p in pairs if p.latency_ms is not None]

    return {
        "mock_run": is_mock,
        "total_queries": total,
        "verdict_counts": counts,
        "accuracy_strict": (n_correct / total) if total and not is_mock else None,
        "accuracy_lenient": (
            (n_correct + 0.5 * n_partial) / total if total and not is_mock else None
        ),
        "no_answer_rate": (counts.get("no_answer", 0) / total) if total else None,
        "parse_failures": sum(1 for v in verdicts if not v.parse_ok),
        "latency_ms": {
            "n": len(latencies),
            "mean": statistics.fmean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "unmatched_agent_segments": len(unmatched or []),
        "meta": meta or {},
        "per_pair": [
            {
                "pair_id": p.pair_id,
                "query": p.query_text,
                "answer_asr": p.response_text,
                "reference": p.reference_text,
                "verdict": by_id[p.pair_id].verdict,
                "confidence": by_id[p.pair_id].confidence,
                "reason": by_id[p.pair_id].reason,
                "latency_ms": p.latency_ms,
                "status": p.status,
            }
            for p in pairs
            if p.pair_id in by_id
        ],
    }


def _fmt(x, spec="{:.1f}", dash="—"):
    return dash if x is None else spec.format(x)


def render_markdown(summary: dict[str, Any]) -> str:
    L: list[str] = ["# Moshi agent evaluation report", ""]

    if summary["mock_run"]:
        L += [
            "> **⚠️  MOCK RUN — NOT A REAL EVALUATION.**",
            "> The judge backend was `mock`; no Qwen model was loaded and no",
            "> answer was actually assessed. Accuracy is intentionally blank.",
            "",
        ]

    meta = summary.get("meta", {})
    if meta:
        L += ["## Run", ""]
        L += [f"- **{k}**: `{v}`" for k, v in meta.items()]
        L.append("")

    lat = summary["latency_ms"]
    L += [
        "## Headline", "",
        f"- Queries addressed to the agent: **{summary['total_queries']}**",
        f"- Strict accuracy (correct only): **{_fmt(summary['accuracy_strict'], '{:.1%}')}**",
        f"- Lenient accuracy (partial = 0.5): **{_fmt(summary['accuracy_lenient'], '{:.1%}')}**",
        f"- No-answer rate: **{_fmt(summary['no_answer_rate'], '{:.1%}')}**",
        f"- Response latency (ms): median {_fmt(lat['median'])}, "
        f"mean {_fmt(lat['mean'])}, range {_fmt(lat['min'])}–{_fmt(lat['max'])}",
        f"- Agent segments matched to no query: **{summary['unmatched_agent_segments']}**",
        f"- Judge output parse failures: **{summary['parse_failures']}**",
        "",
        "## Verdict breakdown", "",
        "| verdict | n |", "| --- | ---: |",
    ]
    for k, v in sorted(summary["verdict_counts"].items(), key=lambda kv: -kv[1]):
        L.append(f"| {k} | {v} |")

    L += ["", "## Per-question detail", ""]
    for row in summary["per_pair"]:
        L += [
            f"### {row['pair_id']} — `{row['verdict']}`"
            + (f" (conf {row['confidence']:.2f})" if row["confidence"] is not None else ""),
            "",
            f"- **Q:** {row['query']}",
            f"- **A (ASR):** {row['answer_asr'] or '_(silent)_'}",
        ]
        if row["reference"]:
            L.append(f"- **Reference:** {row['reference']}")
        L += [
            f"- **Latency:** {_fmt(row['latency_ms'])} ms",
            f"- **Judge:** {row['reason'] or '—'}",
            "",
        ]

    return "\n".join(L)
