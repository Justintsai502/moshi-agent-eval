# moshi-agent-eval

Evaluate a **Moshi-driven spoken agent** against the human-side ground truth of a
multi-party conversation.

The setup: three time-aligned audio stems (A, B human; C = the agent). Once Moshi
replaces the scripted TTS agent, **track C has audio but no transcript** — so its
answers have to be recovered by ASR and then judged.

```
agent track (.wav)  ──ASR──▶  hypothesis turns  ──┐
                                                  ├─▶ Qwen judge ─▶ verdicts ─▶ report
aligned_script.jsonl (humans) ──▶ agent-directed ─┘
                                   questions
```

## The safety catch (read this first)

**Nothing in this repo loads a model by default.** `judge.backend` defaults to
`mock` and `asr.backend` to `gold`. Both `vllm` / `transformers` / `whisper`
paths refuse to import unless you set:

```bash
export MOSHI_EVAL_ALLOW_HEAVY=1
```

A mock run is tagged `mock_unjudged` in every verdict, `mock_run: true` in the
report JSON, accuracy is left `null`, and the markdown report opens with a
warning banner — a mock run can't be mistaken for a real evaluation.

## Two input layouts

**A. Clip pack** (what `bathrooms/.../phase3_p0_IF_evalset_subset` is — the real case)

```
<pack>/ids.txt
<pack>/clips/<clip_id>/input.wav      # user A+B mix
<pack>/clips/<clip_id>/response.wav   # Moshi output, pad_sec=20 longer
authored_scripts/<clip_id>_qa.json    # authored ground truth, keys "A,0"
```

The authored script has exact text but **no timestamps**; `input.wav` has the
clock. `response.wav` shares that clock (plus a 20 s tail). Stage: `pack`.

**B. Single session** — one `aligned_script.jsonl` with `start`/`end` plus one
agent stem. Stages: `asr`/`pair`/`judge`/`report`/`run`.

## Pairing modes

| mode | how | needs |
|---|---|---|
| `time` (default) | ASR `input.wav`, align authored turns to it by token-F1 to timestamp each question, then match agent bursts on the shared clock | 2 ASR passes |
| `order` | k-th agent burst answers the k-th question | 1 ASR pass |

`order` is cheaper but **not safe on this pack**: `scripts/vad_preflight.py`
measures 36/49 clips whose agent-burst count differs from their question count
(5 clips are silent throughout, others speak 4–7 times), so k-th-to-k-th would
misalign most of the set. Use `time`.

`input.wav` is transcribed only to recover timing — the judge always sees the
authored question text, never the ASR of it.

## Preflight (no models, run this first)

```bash
python3 scripts/vad_preflight.py
```

Energy-VAD burst counts per clip vs. question counts, plus pad and silence
stats, written to `runs/vad_preflight.json`. It tells you what the run will
struggle with before you spend GPU time.

## Local (laptop) — verify the plumbing

```bash
bash scripts/local_dryrun.sh                                  # single-session sample
PYTHONPATH=src python3 -m moshi_eval -c config/pack_offline.yaml pack   # all 49 clips
```

Both load zero models: `asr.backend: gold` substitutes the authored text on a
shared synthetic clock, `judge.backend: mock` emits `mock_unjudged`. This
exercises discovery, alignment, pairing, prompt building and reporting.

```bash
python3 tests/test_offline.py && python3 tests/test_pack.py
```

## Server — the real run

```bash
pip install -r requirements-server.txt
bash scripts/pack_server_run.sh config/pack_server.yaml pack_phase3_p0   # clip pack
bash scripts/server_run.sh config/server_moshi.yaml moshi_run01          # single session
```

Stages run separately (`asr` → `pair` → `judge` → `report`) so a judge crash
doesn't discard the ASR pass. Re-run one stage with:

```bash
PYTHONPATH=src MOSHI_EVAL_ALLOW_HEAVY=1 python3 -m moshi_eval \
  -c config/server_moshi.yaml -s judge.backend=vllm judge
```

## Stages

| stage | reads | writes | cost |
|---|---|---|---|
| `asr` | agent `.wav` | `asr.jsonl` | GPU |
| `pair` | `asr.jsonl` + ground truth | `pairs.jsonl`, `prompts.jsonl`, `unmatched_agent_segments.jsonl` | free |
| `judge` | `pairs.jsonl` | `verdicts.jsonl` | GPU |
| `report` | `pairs` + `verdicts` | `report.json`, `report.md`, `config.used.json` | free |

## How questions are found

A ground-truth turn counts as a question to the agent when
`function == "speech"` **and** the selector matches:

- `addressing` (default) — `addressing` contains the agent speaker id
- `wakeword` — text matches `AI Agent` / `hey agent` / `assistant`
- `any` / `both` — combine the two

Backchannels are never queries. **Row order in the jsonl is not time order** —
overlapping turns appear out of sequence (in the sample `g006` precedes `g005`
but starts later), so everything is re-sorted by `start`.

## How answers are matched

An ASR group answers a query if it starts within
`[query_end − overlap_tolerance_s, min(query_end + response_window_s, next_query_start + tol)]`.
Consecutive ASR segments closer than `merge_gap_s` merge into one answer. The
negative lower bound allows barge-in — Moshi is full-duplex and may start before
the human finishes.

Unmatched agent speech is written to `unmatched_agent_segments.jsonl`. **Check
that file**: it's either unprompted agent output (interesting) or a pairing
miss (a bug).

## Verdicts

`correct` · `partially_correct` · `incorrect` · `not_responsive` ·
`no_answer` · `unintelligible` · `mock_unjudged`

`no_answer` and `unintelligible` for empty spans are decided by rule, without
burning a judge call.

## What the numbers can and cannot say

- `no_answer` with `time_known: false` means **alignment failed**, not that the
  agent was silent. The two are never merged.
- `count_mismatch` (order mode only) marks clips where burst count ≠ question
  count, so an off-by-one match is possible.
- A run configured with `judge.backend: mock` reports `accuracy: null` even
  when every pair was settled by rule and the model was never called.

## Reference-free vs reference-based

The authored scripts **do** contain the agent's intended answers (the `C,*`
turns), so the clip pack can be judged reference-based — `config/pack_server.yaml`
sets `use_reference: true`. Only the single-session Moshi layout, where no
agent transcript exists at all, needs `use_reference: false`.

Before trusting Moshi numbers, run `config/server_calibration.yaml`: the real
judge, reference-free, over the scripted TTS audio whose answers you already
know. Anything it gets wrong there is a judge or ASR problem, not an agent one.

## Preparing a Moshi run

```
data/moshi_run01/
  aligned_script.jsonl   # human turns only (A, B) — same schema as the sample
  agent_track.wav        # Moshi output stem, same session clock as the humans
```

The agent stem **must share the session timeline** with the human tracks —
pairing is purely timestamp-based. If Moshi is recorded on its own clock,
offset-correct it first.

## Known data quirk

In `aligned_script.jsonl`, top-level `gap_ms` disagrees with `rel.gap_ms` for
overlapping turns (`g005`: `-7081.6` vs `+77.0`). This pipeline reads neither —
it works from `start` / `end` — but anything else consuming these files should
be aware.
