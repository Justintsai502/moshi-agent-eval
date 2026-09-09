"""Qwen-as-judge.

Backends
--------
mock          default; NO model is loaded. Emits `mock_unjudged` verdicts so a
              mock run can never be mistaken for a real evaluation.
vllm          offline batch on the server. Fastest for a whole session.
transformers  plain HF generate, one prompt at a time. Fallback.
openai        OpenAI-compatible HTTP endpoint (e.g. `vllm serve`). Loads
              nothing locally, so it is safe to call from a laptop -- but it
              does send the transcript to whatever host you point it at.

Heavy local backends additionally require MOSHI_EVAL_ALLOW_HEAVY=1.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from .prompts import build_messages
from .schema import VERDICTS, QAPair, Verdict

HEAVY_ENV = "MOSHI_EVAL_ALLOW_HEAVY"


class HeavyModelBlocked(RuntimeError):
    pass


def _guard(what: str) -> None:
    if os.environ.get(HEAVY_ENV) != "1":
        raise HeavyModelBlocked(
            f"Refusing to load {what} because {HEAVY_ENV}!=1.\n"
            "This is the laptop safety catch -- a 32B judge will not run here.\n"
            f"On the server: export {HEAVY_ENV}=1"
        )


@dataclass
class JudgeConfig:
    backend: str = "mock"
    model: str = "Qwen/Qwen3-32B"
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    use_reference: bool = True
    # vllm
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 8192
    dtype: str = "bfloat16"
    # openai-compatible
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "OPENAI_API_KEY"
    # qwen3 thinking mode: keep it off, we want terse JSON
    extra_body: dict = field(default_factory=lambda: {"chat_template_kwargs": {"enable_thinking": False}})


# --- output parsing ----------------------------------------------------------

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_verdict(raw: str, pair_id: str, cfg: JudgeConfig) -> Verdict:
    cleaned = _THINK_RE.sub("", raw or "").strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()

    m = _JSON_RE.search(cleaned)
    base = Verdict(
        pair_id=pair_id,
        verdict="unintelligible",
        judge_model=cfg.model,
        backend=cfg.backend,
        raw_output=raw or "",
    )
    if not m:
        base.parse_ok = False
        base.reason = "judge output contained no JSON object"
        return base

    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        base.parse_ok = False
        base.reason = f"judge output was not valid JSON: {exc}"
        return base

    v = str(obj.get("verdict", "")).strip().lower()
    if v not in VERDICTS:
        base.parse_ok = False
        base.reason = f"judge returned unknown verdict {v!r}"
        return base

    conf = obj.get("confidence")
    base.verdict = v
    base.confidence = float(conf) if isinstance(conf, (int, float)) else None
    base.answer_summary = str(obj.get("answer_summary", ""))
    base.reason = str(obj.get("reason", ""))
    base.parse_ok = True
    return base


# --- backends ----------------------------------------------------------------

class MockBackend:
    """Loads nothing. Returns a structurally valid but explicitly unjudged result."""

    is_mock = True

    def __init__(self, cfg: JudgeConfig):
        self.cfg = cfg

    def generate(self, messages_batch):
        return [
            json.dumps(
                {
                    "verdict": "mock_unjudged",
                    "confidence": 0.0,
                    "answer_summary": "",
                    "reason": "MOCK BACKEND -- no model was run. Not a real evaluation.",
                }
            )
            for _ in messages_batch
        ]


class VLLMBackend:
    is_mock = False

    def __init__(self, cfg: JudgeConfig):
        _guard(f"vLLM {cfg.model}")
        from vllm import LLM, SamplingParams  # noqa: PLC0415
        from transformers import AutoTokenizer  # noqa: PLC0415

        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(cfg.model)
        self.llm = LLM(
            model=cfg.model,
            tensor_parallel_size=cfg.tensor_parallel_size,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            max_model_len=cfg.max_model_len,
            dtype=cfg.dtype,
        )
        self.sampling = SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens,
        )

    def generate(self, messages_batch):
        prompts = [
            self.tok.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            for m in messages_batch
        ]
        outs = self.llm.generate(prompts, self.sampling)
        return [o.outputs[0].text for o in outs]


class TransformersBackend:
    is_mock = False

    def __init__(self, cfg: JudgeConfig):
        _guard(f"transformers {cfg.model}")
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self.cfg = cfg
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(cfg.model)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model, torch_dtype=getattr(torch, cfg.dtype), device_map="auto"
        )
        self.model.eval()

    def generate(self, messages_batch):
        outs = []
        for messages in messages_batch:
            text = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self.tok([text], return_tensors="pt").to(self.model.device)
            with self.torch.no_grad():
                gen = self.model.generate(
                    **inputs,
                    max_new_tokens=self.cfg.max_new_tokens,
                    do_sample=self.cfg.temperature > 0,
                    temperature=self.cfg.temperature or None,
                    top_p=self.cfg.top_p,
                )
            new = gen[0][inputs["input_ids"].shape[-1]:]
            outs.append(self.tok.decode(new, skip_special_tokens=True))
        return outs


class OpenAICompatBackend:
    """Talks to an already-running server; loads no weights locally."""

    is_mock = False

    def __init__(self, cfg: JudgeConfig):
        from openai import OpenAI  # noqa: PLC0415

        self.cfg = cfg
        self.client = OpenAI(
            base_url=cfg.base_url,
            api_key=os.environ.get(cfg.api_key_env, "EMPTY"),
        )

    def generate(self, messages_batch):
        outs = []
        for messages in messages_batch:
            resp = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=messages,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                max_tokens=self.cfg.max_new_tokens,
                extra_body=self.cfg.extra_body,
            )
            outs.append(resp.choices[0].message.content or "")
        return outs


_BACKENDS = {
    "mock": MockBackend,
    "vllm": VLLMBackend,
    "transformers": TransformersBackend,
    "openai": OpenAICompatBackend,
}


def build_backend(cfg: JudgeConfig):
    try:
        return _BACKENDS[cfg.backend](cfg)
    except KeyError:
        raise ValueError(
            f"unknown judge backend {cfg.backend!r}; "
            f"pick one of {sorted(_BACKENDS)}"
        ) from None


# --- driver ------------------------------------------------------------------

def judge_pairs(pairs: list[QAPair], cfg: JudgeConfig) -> list[Verdict]:
    """Judge every answered pair. Silent pairs are settled without a model."""
    verdicts: list[Verdict] = []
    to_judge: list[QAPair] = []

    for p in pairs:
        if p.status == "no_response":
            verdicts.append(
                Verdict(
                    pair_id=p.pair_id,
                    verdict="no_answer",
                    confidence=1.0,
                    reason="No agent speech fell inside the response window.",
                    judge_model=cfg.model,
                    backend="rule",
                )
            )
        elif p.status == "empty_asr":
            verdicts.append(
                Verdict(
                    pair_id=p.pair_id,
                    verdict="unintelligible",
                    confidence=1.0,
                    reason="ASR returned an empty transcript for the response span.",
                    judge_model=cfg.model,
                    backend="rule",
                )
            )
        else:
            to_judge.append(p)

    if to_judge:
        backend = build_backend(cfg)
        batch = [build_messages(p, cfg.use_reference) for p in to_judge]
        raws = backend.generate(batch)
        for p, raw in zip(to_judge, raws):
            v = parse_verdict(raw, p.pair_id, cfg)
            v.is_mock = getattr(backend, "is_mock", False)
            verdicts.append(v)

    order = {p.pair_id: i for i, p in enumerate(pairs)}
    verdicts.sort(key=lambda v: order[v.pair_id])
    return verdicts
