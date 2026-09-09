"""ASR over the agent audio track.

Every heavy import lives inside its backend function, so `import moshi_eval.asr`
stays cheap on a laptop. Nothing loads a model unless you explicitly pick a
real backend AND set MOSHI_EVAL_ALLOW_HEAVY=1.
"""

from __future__ import annotations

import os
import wave
from dataclasses import dataclass

from .schema import Utterance

HEAVY_ENV = "MOSHI_EVAL_ALLOW_HEAVY"


class HeavyModelBlocked(RuntimeError):
    pass


def _guard(what: str) -> None:
    if os.environ.get(HEAVY_ENV) != "1":
        raise HeavyModelBlocked(
            f"Refusing to load {what} because {HEAVY_ENV}!=1.\n"
            "This is the laptop safety catch. On the server, export "
            f"{HEAVY_ENV}=1 first."
        )


@dataclass
class ASRConfig:
    backend: str = "gold"             # gold | faster_whisper | whisper | mock
    model: str = "large-v3"
    language: str | None = "en"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 5
    vad_filter: bool = True
    min_segment_s: float = 0.15


def audio_duration(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


# --- backends ----------------------------------------------------------------

def _asr_faster_whisper(audio_path: str, cfg: ASRConfig) -> list[Utterance]:
    _guard(f"faster-whisper {cfg.model}")
    from faster_whisper import WhisperModel  # noqa: PLC0415  (lazy on purpose)

    model = WhisperModel(cfg.model, device=cfg.device, compute_type=cfg.compute_type)
    segments, _info = model.transcribe(
        audio_path,
        language=cfg.language,
        beam_size=cfg.beam_size,
        vad_filter=cfg.vad_filter,
        word_timestamps=False,
    )
    out: list[Utterance] = []
    for i, seg in enumerate(segments):
        text = (seg.text or "").strip()
        if not text or (seg.end - seg.start) < cfg.min_segment_s:
            continue
        out.append(
            Utterance(
                utt_id=f"asr{i:04d}",
                speaker="AGENT",
                start=float(seg.start),
                end=float(seg.end),
                text=text,
                source="asr",
                asr_confidence=(
                    float(getattr(seg, "avg_logprob", 0.0))
                    if hasattr(seg, "avg_logprob") else None
                ),
            )
        )
    return out


def _asr_whisper(audio_path: str, cfg: ASRConfig) -> list[Utterance]:
    _guard(f"openai-whisper {cfg.model}")
    import whisper  # noqa: PLC0415

    model = whisper.load_model(cfg.model, device=cfg.device)
    result = model.transcribe(audio_path, language=cfg.language, verbose=False)
    out: list[Utterance] = []
    for i, seg in enumerate(result.get("segments", [])):
        text = (seg.get("text") or "").strip()
        if not text or (seg["end"] - seg["start"]) < cfg.min_segment_s:
            continue
        out.append(
            Utterance(
                utt_id=f"asr{i:04d}",
                speaker="AGENT",
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=text,
                source="asr",
                asr_confidence=seg.get("avg_logprob"),
            )
        )
    return out


def _asr_from_gold(gold_agent_turns: list[Utterance]) -> list[Utterance]:
    """Oracle 'ASR': reuse the scripted agent text.

    Lets the pairing + judging stages be exercised end to end with no model.
    Only usable on the scripted TTS sample, never on real Moshi output.
    """
    return [
        Utterance(
            utt_id=f"asr_gold_{u.utt_id}",
            speaker="AGENT",
            start=u.start,
            end=u.end,
            text=u.text,
            source="asr",
        )
        for u in gold_agent_turns
    ]


def _asr_mock(audio_path: str, cfg: ASRConfig) -> list[Utterance]:
    """One placeholder segment spanning the file. Structure only, no content."""
    dur = audio_duration(audio_path)
    return [
        Utterance(
            utt_id="asr0000",
            speaker="AGENT",
            start=0.0,
            end=dur,
            text="<mock asr: no transcription performed>",
            source="asr",
        )
    ]


def transcribe(
    audio_path: str,
    cfg: ASRConfig,
    gold_agent_turns: list[Utterance] | None = None,
) -> list[Utterance]:
    if cfg.backend == "gold":
        if not gold_agent_turns:
            raise ValueError(
                "asr.backend='gold' needs agent rows in the ground-truth jsonl. "
                "A real Moshi run has none -- use 'faster_whisper' there."
            )
        segs = _asr_from_gold(gold_agent_turns)
    elif cfg.backend == "faster_whisper":
        segs = _asr_faster_whisper(audio_path, cfg)
    elif cfg.backend == "whisper":
        segs = _asr_whisper(audio_path, cfg)
    elif cfg.backend == "mock":
        segs = _asr_mock(audio_path, cfg)
    else:
        raise ValueError(f"unknown asr backend: {cfg.backend!r}")

    segs.sort(key=lambda u: u.start)
    return segs
