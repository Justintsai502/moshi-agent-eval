#!/usr/bin/env python3
"""Predict, without any model, how order-based pairing will behave on a pack.

Counts energy bursts in each clip's response.wav and compares that to the
number of questions in the authored script. Anything unequal is a clip where
the k-th-burst-to-k-th-question match may be wrong, so you learn the scale of
the problem *before* spending GPU time on ASR + judge.

    python3 scripts/vad_preflight.py [pack_dir] [scripts_dir]
"""

from __future__ import annotations

import json
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from moshi_eval.authored import load_authored_script  # noqa: E402

WIN_S = 0.05
THRESH = 0.01          # RMS floor for "speech"
MERGE_GAP_S = 0.8      # must match pairing.merge_gap_s
MIN_BURST_S = 0.35     # shorter than this is a click/breath, not an answer


def bursts(path: str) -> list[tuple[float, float]]:
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    x = x.astype(np.float32) / 32768.0
    n = max(1, int(sr * WIN_S))
    m = len(x) // n
    if m == 0:
        return []
    rms = np.sqrt((x[: m * n].reshape(m, n) ** 2).mean(axis=1))
    active = rms > THRESH

    spans: list[list[float]] = []
    start = None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            spans.append([start * WIN_S, i * WIN_S])
            start = None
    if start is not None:
        spans.append([start * WIN_S, len(active) * WIN_S])

    merged: list[list[float]] = []
    for s in spans:
        if merged and s[0] - merged[-1][1] <= MERGE_GAP_S:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [(a, b) for a, b in merged if b - a >= MIN_BURST_S]


def main() -> int:
    root = os.path.join(os.path.dirname(__file__), "..")
    pack = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        root, "bathrooms/kcire/mpmoshi-ifami/packs/phase3_p0_IF_evalset_subset")
    scripts = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "authored_scripts")

    ids = [l.strip() for l in open(os.path.join(pack, "ids.txt")) if l.strip()]
    rows, mismatch, silent = [], [], []

    for cid in ids:
        resp = os.path.join(pack, "clips", cid, "response.wav")
        inp = os.path.join(pack, "clips", cid, "input.wav")
        scr = os.path.join(scripts, f"{cid}_qa.json")
        if not (os.path.exists(resp) and os.path.exists(scr)):
            continue

        b = bursts(resp)
        nq = len(load_authored_script(scr, cid).queries)
        with wave.open(inp) as w:
            din = w.getnframes() / w.getframerate()
        with wave.open(resp) as w:
            dre = w.getnframes() / w.getframerate()

        row = {
            "clip_id": cid, "n_bursts": len(b), "n_questions": nq,
            "input_s": round(din, 2), "response_s": round(dre, 2),
            "pad_s": round(dre - din, 2),
            "speech_s": round(sum(e - s for s, e in b), 2),
            "bursts": [[round(s, 2), round(e, 2)] for s, e in b],
            "after_pad_start": sum(1 for s, _ in b if s >= din),
        }
        rows.append(row)
        if len(b) != nq:
            mismatch.append(row)
        if not b:
            silent.append(cid)

    n = len(rows)
    print(f"clips analysed        : {n}")
    print(f"burst count == n_questions : {n - len(mismatch)}/{n}")
    print(f"predicted count_mismatch   : {len(mismatch)}/{n} "
          f"({100 * len(mismatch) / n:.0f}%)" if n else "")
    print(f"completely silent response : {len(silent)}")
    pads = {r["pad_s"] for r in rows}
    print(f"pad_s values          : {sorted(pads)}")
    after = sum(r["after_pad_start"] for r in rows)
    print(f"bursts beginning inside the 20s pad tail : {after}")

    if mismatch:
        print("\nclips whose burst count != question count:")
        for r in mismatch[:20]:
            print(f"  {r['clip_id']:52s} bursts={r['n_bursts']} q={r['n_questions']} "
                  f"spans={r['bursts']}")
        if len(mismatch) > 20:
            print(f"  ... and {len(mismatch) - 20} more")

    out = os.path.join(root, "runs", "vad_preflight.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nper-clip detail -> {os.path.relpath(out, root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
