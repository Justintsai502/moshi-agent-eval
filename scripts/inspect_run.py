#!/usr/bin/env python3
"""Show what actually happened in a run, question by question.

    python3 scripts/inspect_run.py runs/pack_smoke3

Answers the two questions a bad report leaves open:
  - did the question get a timestamp at all (time_known)?
  - if it did, what did the agent track say inside the response window?
"""

from __future__ import annotations

import json
import os
import sys


def rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main() -> int:
    run = sys.argv[1] if len(sys.argv) > 1 else "runs/pack_smoke3"
    pairs = rows(os.path.join(run, "pairs.jsonl"))
    asr = rows(os.path.join(run, "asr.jsonl"))
    verdicts = {v["pair_id"]: v for v in rows(os.path.join(run, "verdicts.jsonl"))}
    if not pairs:
        print(f"no pairs.jsonl under {run}")
        return 1

    by_clip: dict[str, list[dict]] = {}
    for p in pairs:
        by_clip.setdefault(p["clip_id"], []).append(p)

    n_untimed = sum(1 for p in pairs if not p.get("time_known", True))
    n_silent = sum(
        1 for p in pairs
        if p["status"] == "no_response" and p.get("time_known", True)
    )

    for cid, ps in by_clip.items():
        print("=" * 78)
        print(f"CLIP {cid}")
        resp = [a for a in asr if a.get("clip_id") == cid and a.get("track") == "response"]
        inp = [a for a in asr if a.get("clip_id") == cid and a.get("track") == "input"]

        print(f"\n  input.wav ASR ({len(inp)} seg):")
        for a in inp:
            print(f"    {a['start']:7.2f}-{a['end']:7.2f}  {a['text'][:68]}")
        print(f"\n  response.wav ASR ({len(resp)} seg)  <- the agent:")
        for a in resp:
            print(f"    {a['start']:7.2f}-{a['end']:7.2f}  {a['text'][:68]}")

        for p in ps:
            v = verdicts.get(p["pair_id"], {})
            when = (
                f"[{p['query_start']:.2f}-{p['query_end']:.2f}]"
                if p.get("query_start") is not None else "[NO TIME RECOVERED]"
            )
            print(f"\n  --- {p['pair_id']}  {when}  status={p['status']}"
                  f"  time_known={p.get('time_known', True)}")
            print(f"      Q      : {p['query_text'][:70]}")
            print(f"      REF    : {(p.get('reference_text') or '-')[:70]}")
            print(f"      AGENT  : {p['response_text'][:70] or '(nothing matched)'}")
            if v:
                print(f"      VERDICT: {v['verdict']}  {v.get('reason','')[:60]}")
        print()

    print("=" * 78)
    print(f"questions total          : {len(pairs)}")
    print(f"  no timestamp recovered : {n_untimed}   <- alignment failed")
    print(f"  timed but agent silent : {n_silent}   <- nothing in the window")
    left = rows(os.path.join(run, "unmatched_agent_segments.jsonl"))
    print(f"agent speech matched to nothing : {len(left)}")
    for a in left[:10]:
        print(f"    {a.get('clip_id','')} {a['start']:7.2f}-{a['end']:7.2f} {a['text'][:50]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
