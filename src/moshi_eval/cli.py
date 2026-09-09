"""Command line entry point.

    python -m moshi_eval --config config/sample_offline.yaml run

Stages can also be run one at a time (asr / pair / judge / report) so the
expensive ones happen on the server and the cheap ones happen locally.
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import load_config
from .pack import run_pack
from .pipeline import run_all, stage_asr, stage_judge, stage_pair, stage_report

STAGES = {
    "asr": lambda cfg: stage_asr(cfg) and None,
    "pair": lambda cfg: stage_pair(cfg) and None,
    "judge": lambda cfg: stage_judge(cfg) and None,
    "report": stage_report,
    "run": run_all,
    "pack": run_pack,
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="moshi_eval", description=__doc__)
    ap.add_argument("stage", choices=sorted(STAGES), help="which stage to run")
    ap.add_argument("-c", "--config", help="path to a YAML config")
    ap.add_argument(
        "-s", "--set", dest="overrides", action="append", default=[],
        metavar="KEY=VALUE",
        help="override a config key, e.g. -s judge.backend=vllm -s run_name=exp1",
    )
    ap.add_argument(
        "--print-prompt", type=int, metavar="N",
        help="after 'pair', print the Nth judge prompt (1-based) and exit",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)

    if cfg.judge.backend == "mock" and args.stage in ("judge", "run"):
        print(
            "[note] judge.backend=mock -- no Qwen model will be loaded and no "
            "answer will actually be judged.\n"
            "       For a real run on the server: -s judge.backend=vllm "
            "(and export MOSHI_EVAL_ALLOW_HEAVY=1)",
            file=sys.stderr,
        )

    result = STAGES[args.stage](cfg)

    if args.print_prompt is not None and args.stage in ("pair", "run"):
        from .pipeline import _p
        from .schema import read_jsonl
        rows = list(read_jsonl(_p(cfg, "prompts.jsonl")))
        i = args.print_prompt - 1
        if not 0 <= i < len(rows):
            print(f"only {len(rows)} prompts available", file=sys.stderr)
            return 1
        for msg in rows[i]["messages"]:
            print(f"\n===== {msg['role'].upper()} =====\n{msg['content']}")
        return 0

    if isinstance(result, dict):
        slim = {k: v for k, v in result.items() if k != "per_pair"}
        print(json.dumps(slim, ensure_ascii=False, indent=2))
    print(f"\nartefacts -> {cfg.out_dir}/{cfg.run_name}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
