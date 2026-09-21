"""Mechanical checks that keep interpretations subordinate to source bytes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import math
from typing import Iterable

from .corpus import CorpusStore
from .models import BindingValue, WitnessRow


@dataclass(frozen=True, slots=True)
class VerificationResult:
    accepted: bool
    errors: tuple[str, ...] = ()


class MechanicalVerifier:
    VERIFIER_VERSION = "mechanical-verifier-0.2.0"

    def verify(
        self,
        row: WitnessRow,
        store: CorpusStore,
        *,
        principals: Iterable[str] = (),
    ) -> VerificationResult:
        errors: list[str] = []
        try:
            document = store.get_document(row.source_id, row.source_version)
        except KeyError:
            return VerificationResult(False, ("source version no longer exists",))
        principal_set = set(principals)
        if document.permissions and not principal_set.intersection(document.permissions):
            errors.append("source is not authorized for these principals")
        if document.content_hash != row.source_hash:
            errors.append("source hash changed")
        if row.start_offset < 0 or row.end_offset > len(document.raw_content):
            errors.append("witness offsets are outside the source")
        elif document.raw_content[row.start_offset : row.end_offset] != row.quote:
            errors.append("quote is not the exact source slice at its offsets")
        for name, binding in row.bindings.items():
            errors.extend(self._verify_binding(name, binding, row, document.raw_content))
        return VerificationResult(not errors, tuple(errors))

    def _verify_binding(
        self,
        name: str,
        binding: BindingValue,
        row: WitnessRow,
        raw_content: str,
    ) -> list[str]:
        errors: list[str] = []
        if not (row.start_offset <= binding.start_offset <= binding.end_offset <= row.end_offset):
            errors.append(f"binding {name} lies outside the witness quote")
            return errors
        if raw_content[binding.start_offset : binding.end_offset] != binding.surface:
            errors.append(f"binding {name} surface does not match source offsets")
        kind = binding.value_type.casefold()
        surface = binding.surface.strip()
        try:
            if kind in {"int", "integer"} and int(surface.replace(",", "")) != binding.value:
                errors.append(f"binding {name} integer differs from source")
            elif kind in {"float", "number", "decimal"}:
                parsed = float(surface.replace(",", ""))
                if not isinstance(binding.value, (int, float)) or not math.isclose(
                    parsed, float(binding.value), rel_tol=1e-12, abs_tol=0.0
                ):
                    errors.append(f"binding {name} number differs from source")
            elif kind == "date":
                date.fromisoformat(surface)
            elif kind == "datetime":
                datetime.fromisoformat(surface[:-1] + "+00:00" if surface.endswith("Z") else surface)
            elif kind in {"bool", "boolean"}:
                expected = surface.casefold() in {"true", "yes"}
                if surface.casefold() not in {"true", "false", "yes", "no"} or binding.value is not expected:
                    errors.append(f"binding {name} boolean differs from source")
            elif str(binding.value).strip() != surface:
                errors.append(f"binding {name} interpretation is not an exact source-backed string")
        except (ValueError, TypeError, OverflowError):
            errors.append(f"binding {name} cannot be mechanically validated as {kind}")
        return errors
