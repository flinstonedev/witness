"""Demand-compiled, per-program/per-shard witness view cache."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import threading

from .models import EvidenceRole, LocalEvaluation, SourceBlock, stable_hash


@dataclass(frozen=True, slots=True)
class CacheKey:
    program_hash: str
    predicate_hash: str
    shard_hash: str
    evaluator_version: str
    role: EvidenceRole

    @classmethod
    def create(
        cls,
        *,
        program_hash: str,
        predicate_hash: str,
        block: SourceBlock,
        evaluator_version: str,
        role: EvidenceRole,
    ) -> "CacheKey":
        # Include source identity because identical bytes in two documents need
        # distinct citations even though their semantic evaluation is reusable.
        shard_hash = stable_hash(
            [
                block.document_id,
                block.document_version,
                block.start_offset,
                block.end_offset,
                block.block_hash,
            ]
        )
        return cls(program_hash, predicate_hash, shard_hash, evaluator_version, role)

    @property
    def digest(self) -> str:
        return stable_hash(
            [
                self.program_hash,
                self.predicate_hash,
                self.shard_hash,
                self.evaluator_version,
                self.role.value,
            ]
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "program_hash": self.program_hash,
            "predicate_hash": self.predicate_hash,
            "shard_hash": self.shard_hash,
            "evaluator_version": self.evaluator_version,
            "role": self.role.value,
        }


class MaterializedWitnessCache:
    """Thread-safe optional disk cache for local evidence evaluations."""

    CACHE_VERSION = "witness-cache-0.2.0"

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        if self.path is not None:
            self.path.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, LocalEvaluation] = {}
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def _path_for(self, key: CacheKey) -> Path:
        assert self.path is not None
        # Prefix sharding avoids huge single directories for maintained views.
        directory = self.path / key.digest[:2]
        return directory / f"{key.digest}.json"

    def get(self, key: CacheKey) -> LocalEvaluation | None:
        with self._lock:
            cached = self._memory.get(key.digest)
            if cached is not None:
                self.hits += 1
                return cached
            if self.path is not None:
                path = self._path_for(key)
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if payload.get("cache_version") != self.CACHE_VERSION:
                        raise ValueError("cache version mismatch")
                    if payload.get("key") != key.to_dict():
                        raise ValueError("cache key mismatch")
                    cached = LocalEvaluation.from_dict(payload["evaluation"])
                except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                    cached = None
                if cached is not None:
                    self._memory[key.digest] = cached
                    self.hits += 1
                    return cached
            self.misses += 1
            return None

    def put(self, key: CacheKey, evaluation: LocalEvaluation) -> None:
        with self._lock:
            self._memory[key.digest] = evaluation
            if self.path is None:
                return
            path = self._path_for(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            payload = {
                "cache_version": self.CACHE_VERSION,
                "key": key.to_dict(),
                "evaluation": evaluation.to_dict(),
            }
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            os.replace(temporary, path)

    def reset_stats(self) -> None:
        with self._lock:
            self.hits = 0
            self.misses = 0

