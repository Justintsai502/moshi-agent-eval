# moshi-agent-eval

Evaluates the answers a Moshi-driven AI agent gives inside a multi-party
conversation. The agent's audio track is transcribed with ASR, each answer is
matched to the question it responds to, and Qwen3-32B judges it against the
scripted reference answer.

```
aligned_script.jsonl ──▶ questions to the agent (text + start/end)
                                   │
response.wav ──ASR──▶ agent speech ─┴─▶ match by time ─▶ Qwen judge ─▶ report
```

## Data

```
<scripts_root>/<clip_id>/aligned_script.jsonl    timed ground truth, every speaker
<eval_pack>/ids.txt                              clips to evaluate
<eval_pack>/clips/<clip_id>/input.wav            users A+B+C, fed to the model
<eval_pack>/clips/<clip_id>/response.wav         model output, 20 s longer
```

The clock in `aligned_script.jsonl` is the clock of `input.wav`, and
`response.wav` starts at the same origin. Question timestamps are therefore
read straight from the script.

## How it works

**Agent speaker.** Detected from the data: whoever the "AI Agent, …" turns
address (`D` in the 4-speaker set).

**Questions.** Human turns whose `addressing` contains the agent. Both
`speech` and `continuer` turns count; backchannels are ignored throughout.

**Reference answer.** The scripted agent turn that follows the question,
bounded by the next question.

**ASR.** faster-whisper `large-v3` on `response.wav` only. The human side is
never transcribed — its text is already in the script.

**Matching.** An agent segment answers a question if it starts within
`[q_end − 1.0 s, min(q_end + 12 s, next_q_start + 1.0 s)]`. Segments less than
0.8 s apart are merged into one answer. Filler-only segments (`uh`, `um`, …)
are dropped; one-word answers are kept.

**Judging.** Qwen3-32B via vLLM gets the preceding human speech, the question,
the ASR'd answer and the reference, and returns JSON:
`correct` · `partially_correct` · `incorrect` · `not_responsive` ·
`unintelligible`. Questions with no agent speech in the window are marked
`no_answer` by rule, without a model call.

## Running

Nothing loads a model unless `MOSHI_EVAL_ALLOW_HEAVY=1` is set.

**Local, no models** — scripted agent text in place of ASR, mock judge:

```bash
PYTHONPATH=src python3 -m moshi_eval -c config/if4_offline.yaml timed
```

**Server:**

```bash
pip install -r requirements-server.txt
module load cuda/12.6
bash scripts/if4_server_run.sh
```

Paths live in `config/if4_server.yaml`; override any key with
`-s key=value`, e.g. `-s judge.model=/path/to/Qwen3-32B/snapshot`.

**Calibration** — scripted agent text in place of ASR, real judge. Every
answer is correct by construction, so this measures the judge and the matching
on their own:

```bash
PYTHONPATH=src MOSHI_EVAL_ALLOW_HEAVY=1 python3 -m moshi_eval \
  -c config/if4_offline.yaml -s judge.backend=vllm timed
```

On the 155-clip IF4 set this scores 310/310.

**Re-judging without re-transcribing:** `-s data.reuse_asr=true` loads
`asr.jsonl` from the previous run of the same `run_name`.

## Output

Written to `runs/<run_name>/`:

| file | contents |
|---|---|
| `pairs.jsonl` | per question: text, reference, agent's ASR'd answer, timing |
| `verdicts.jsonl` | judge decision and reason per question |
| `report.md` / `report.json` | accuracy, verdict counts, no-answer rate, latency |
| `asr.jsonl` | raw ASR segments |
| `prompts.jsonl` | exact prompts sent to the judge |
| `unmatched_agent_segments.jsonl` | agent speech that answered no question |

`python3 scripts/inspect_run.py runs/<run_name>` prints each question next to
what the agent said.

## Other input layouts

- `pack` stage — 3-speaker clip packs whose authored scripts
  (`<clip_id>_qa.json`) have no timestamps; question times are recovered by
  transcribing `input.wav` and aligning the script to it.
- `run` stage — a single long session with one `aligned_script.jsonl` and one
  agent stem.
