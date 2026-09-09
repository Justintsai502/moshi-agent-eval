"""Judge prompt construction.

Two modes:
  reference-free  -- the real case. No script exists for the Moshi track, so
                     the judge rules on world knowledge + responsiveness.
  reference-based -- calibration against the scripted TTS run, where the
                     intended answer is known.
"""

from __future__ import annotations

from .schema import QAPair

SYSTEM_PROMPT = """\
You are a strict evaluator for a spoken-dialogue AI agent.

You will see a short slice of a multi-party conversation and one question that
a human speaker addressed to the AI agent. You will then see what the agent
actually said, as produced by an automatic speech recognizer.

Judging rules:
1. The answer text is ASR output. It may contain transcription errors,
   missing punctuation, and numbers spelled out as words ("one twenty eight"
   for "128"). Judge the MEANING, not the spelling. Do not penalise ASR noise
   that leaves the meaning clear.
2. Judge two things together: factual correctness, and whether it actually
   answers the question that was asked.
3. A confident answer that is factually wrong is "incorrect". An answer that
   is factually fine but ignores the question is "not_responsive".
4. If the transcript is so garbled that you cannot recover any claim, use
   "unintelligible". Do not guess.
5. Reasonable ranges count as correct when the true value falls inside them.

Reply with a single JSON object and nothing else:
{"verdict": "<correct|partially_correct|incorrect|not_responsive|unintelligible>",
 "confidence": <float 0.0-1.0>,
 "answer_summary": "<the agent's claim in one short clause>",
 "reason": "<one or two sentences>"}
"""

_USER_TEMPLATE = """\
## Conversation context (human speakers only)
{context}

## Question addressed to the AI agent
Speaker {speaker}{qwhen}: {question}

## What the agent's audio track said (ASR transcript){timing}
{answer}
{reference}
## Task
Evaluate the agent's answer using the rules you were given. Output only the JSON object.
"""

_REFERENCE_BLOCK = """
## Reference answer (the intended response for this question)
{reference}
Treat this as the gold answer. An agent answer that agrees with it in meaning
is correct even if worded differently.
"""


def build_user_prompt(pair: QAPair, use_reference: bool = True) -> str:
    context = "\n".join(pair.context) if pair.context else "(no prior turns)"

    qwhen = f" at {pair.query_start:.2f}s" if pair.query_start is not None else ""

    if pair.response_start is not None:
        timing = f"\n(spoken {pair.response_start:.2f}s-{pair.response_end:.2f}s"
        timing += (
            f"; {pair.latency_ms:.0f} ms after the question ended)"
            if pair.latency_ms is not None else ")"
        )
    else:
        timing = ""

    answer = pair.response_text.strip() or "(the agent track was silent -- no answer)"

    reference = ""
    if use_reference and pair.reference_text:
        reference = _REFERENCE_BLOCK.format(reference=pair.reference_text.strip())

    return _USER_TEMPLATE.format(
        context=context,
        speaker=pair.query_speaker,
        qwhen=qwhen,
        question=pair.query_text.strip(),
        timing=timing,
        answer=answer,
        reference=reference,
    )


def build_messages(pair: QAPair, use_reference: bool = True) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(pair, use_reference)},
    ]
