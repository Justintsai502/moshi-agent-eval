"""YAML config -> dataclasses, with CLI overrides via dotted keys."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from typing import Any

from .asr import ASRConfig
from .judge import JudgeConfig
from .pairing import PairingConfig


@dataclass
class DataConfig:
    # single-session mode (aligned_script.jsonl + one agent stem)
    ground_truth: str = ""
    agent_audio: str = ""
    query_selector: str = "addressing"   # addressing | wakeword | both | any
    # clip-pack mode (ids.txt + clips/<id>/{input,response}.wav + *_qa.json)
    pack_dir: str = ""
    scripts_dir: str = ""
    limit: int = 0                       # 0 = all clips
    agent_speaker: str = "C"


@dataclass
class Config:
    run_name: str = "default"
    out_dir: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    pairing: PairingConfig = field(default_factory=PairingConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)


_SECTIONS = {"data": DataConfig, "asr": ASRConfig, "pairing": PairingConfig, "judge": JudgeConfig}


def _load_yaml(path: str) -> dict[str, Any]:
    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        raise SystemExit(
            "PyYAML is needed to read config files: pip install pyyaml"
        ) from None
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _coerce(value: str) -> Any:
    for caster in (json.loads,):
        try:
            return caster(value)
        except Exception:
            pass
    return value


def load_config(path: str | None = None, overrides: list[str] | None = None) -> Config:
    raw: dict[str, Any] = _load_yaml(path) if path else {}

    for kv in overrides or []:
        if "=" not in kv:
            raise SystemExit(f"--set expects key=value, got {kv!r}")
        key, val = kv.split("=", 1)
        node = raw
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(val)

    cfg = Config(
        run_name=raw.get("run_name", "default"),
        out_dir=raw.get("out_dir", "runs"),
    )
    for name, klass in _SECTIONS.items():
        section = raw.get(name) or {}
        known = {f.name for f in fields(klass)}
        unknown = set(section) - known
        if unknown:
            raise SystemExit(f"unknown key(s) in [{name}]: {sorted(unknown)}")
        setattr(cfg, name, klass(**section))
    return cfg
