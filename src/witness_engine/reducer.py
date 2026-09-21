"""Deterministic relational reduction over verified witness rows."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from .models import ReducerSpec, ValidationError, WitnessRow, canonical_json


MISSING = object()


def _witness_ids(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    ids: set[str] = set()
    for row in rows:
        value = row.get("__witness_ids", [])
        if isinstance(value, list):
            ids.update(str(item) for item in value)
    return sorted(ids)


def witness_record(row: WitnessRow) -> dict[str, Any]:
    record: dict[str, Any] = {name: binding.value for name, binding in row.bindings.items()}
    record.update(
        {
            "__witness_ids": [row.witness_id],
            "witness_id": row.witness_id,
            "source_id": row.source_id,
            "source_version": row.source_version,
            "start_offset": row.start_offset,
            "end_offset": row.end_offset,
            "quote": row.quote,
            "polarity": row.polarity,
            "modality": row.modality,
            "valid_time": row.valid_time,
            "valid_to": row.valid_to,
            "record_time": row.record_time,
            "role": row.role.value,
        }
    )
    return record


def _get(row: Mapping[str, Any], field: str, default: Any = MISSING) -> Any:
    if field in row:
        return row[field]
    current: Any = row
    for part in field.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _key(row: Mapping[str, Any], fields: Iterable[str]) -> tuple[Any, ...]:
    return tuple(_get(row, field, None) for field in fields)


def _parse_time(value: Any) -> datetime | None:
    if value is None or value is MISSING:
        return None
    normalized = str(value)
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        result = datetime.fromisoformat(normalized)
        if result.tzinfo is None:
            return result.replace(tzinfo=timezone.utc)
        return result.astimezone(timezone.utc)
    except ValueError:
        return None


class DeterministicReducer:
    REDUCER_VERSION = "deterministic-reducer-0.2.1"

    def execute(
        self,
        relations: Mapping[str, Iterable[Mapping[str, Any]] | Iterable[WitnessRow]],
        specs: Iterable[ReducerSpec],
    ) -> dict[str, list[dict[str, Any]]]:
        state: dict[str, list[dict[str, Any]]] = {}
        for name, rows in relations.items():
            materialized: list[dict[str, Any]] = []
            for row in rows:
                materialized.append(witness_record(row) if isinstance(row, WitnessRow) else dict(row))
            state[name] = materialized
        for spec in specs:
            inputs = [state[name] for name in spec.inputs]
            state[spec.output] = self._apply(spec, inputs)
        return state

    def _apply(self, spec: ReducerSpec, inputs: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        operation = spec.operation
        params = dict(spec.params)
        if operation == "FILTER":
            self._arity(spec, inputs, 1)
            conditions = params.get("conditions")
            if conditions is None:
                conditions = [params.get("condition", params)]
            if not isinstance(conditions, list) or not all(isinstance(item, Mapping) for item in conditions):
                raise ValidationError("FILTER conditions must be objects")
            mode = str(params.get("mode", "all")).casefold()
            predicate = all if mode == "all" else any if mode == "any" else None
            if predicate is None:
                raise ValidationError("FILTER mode must be all or any")
            return [
                deepcopy(row)
                for row in inputs[0]
                if predicate(self._condition(row, condition) for condition in conditions)
            ]
        if operation == "JOIN":
            self._arity(spec, inputs, 2)
            return self._join(inputs[0], inputs[1], params)
        if operation == "GROUP_BY":
            self._arity(spec, inputs, 1)
            return self._group_by(inputs[0], params)
        if operation == "COUNT":
            self._arity(spec, inputs, 1)
            name = str(params.get("as", "count"))
            return [{name: len(inputs[0]), "__witness_ids": _witness_ids(inputs[0])}]
        if operation in {"SUM", "MIN", "MAX"}:
            self._arity(spec, inputs, 1)
            field = self._required_string(params, "field")
            name = str(params.get("as", operation.casefold()))
            values = [_get(row, field) for row in inputs[0]]
            values = [value for value in values if value is not MISSING and value is not None]
            if operation == "SUM":
                if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
                    raise ValidationError("SUM accepts only numeric values")
                result = sum(values)
            elif operation == "MIN":
                result = min(values) if values else None
            else:
                result = max(values) if values else None
            return [{name: result, "__witness_ids": _witness_ids(inputs[0])}]
        if operation in {"ARGMAX", "ARGMIN"}:
            self._arity(spec, inputs, 1)
            field = self._required_string(params, "field")
            candidates = [row for row in inputs[0] if _get(row, field) not in (MISSING, None)]
            if not candidates:
                return []
            function = max if operation == "ARGMAX" else min
            selected_value = function(_get(row, field) for row in candidates)
            return [deepcopy(row) for row in candidates if _get(row, field) == selected_value]
        if operation == "SORT":
            self._arity(spec, inputs, 1)
            fields = self._fields(params)
            descending = bool(params.get("descending", False))
            return sorted(
                (deepcopy(row) for row in inputs[0]),
                key=lambda row: tuple(self._sort_atom(_get(row, field, None)) for field in fields),
                reverse=descending,
            )
        if operation == "DEDUPLICATE":
            self._arity(spec, inputs, 1)
            fields = self._fields(params, required=False)
            buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in inputs[0]:
                value = _key(row, fields) if fields else {
                    key: val for key, val in row.items() if not key.startswith("__") and key != "witness_id"
                }
                buckets[canonical_json(value)].append(row)
            output: list[dict[str, Any]] = []
            for key in sorted(buckets):
                first = deepcopy(buckets[key][0])
                first["__witness_ids"] = _witness_ids(buckets[key])
                output.append(first)
            return output
        if operation == "TEMPORAL_RESOLVE":
            self._arity(spec, inputs, 1)
            return self._temporal_resolve(inputs[0], params)
        if operation in {"UNION", "INTERSECTION", "SET_DIFFERENCE"}:
            if len(inputs) < 2:
                raise ValidationError(f"{operation} requires at least two inputs")
            return self._set_operation(operation, inputs, params)
        raise ValidationError(f"unimplemented reducer operation: {operation}")

    @staticmethod
    def _arity(spec: ReducerSpec, inputs: list[Any], expected: int) -> None:
        if len(inputs) != expected:
            raise ValidationError(f"{spec.operation} requires {expected} input relation(s)")

    @staticmethod
    def _required_string(params: Mapping[str, Any], key: str) -> str:
        value = params.get(key)
        if not isinstance(value, str) or not value:
            raise ValidationError(f"reducer param {key} must be a non-empty string")
        return value

    @staticmethod
    def _sort_atom(value: Any) -> tuple[int, Any]:
        if value is None or value is MISSING:
            return (0, "")
        if isinstance(value, bool):
            return (1, int(value))
        if isinstance(value, (int, float)):
            return (2, value)
        if isinstance(value, str):
            return (3, value)
        return (4, canonical_json(value))

    @staticmethod
    def _fields(params: Mapping[str, Any], *, required: bool = True) -> tuple[str, ...]:
        value = params.get("fields")
        if value is None and "field" in params:
            value = [params["field"]]
        if value is None and not required:
            return ()
        if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
            raise ValidationError("fields must be a non-empty list of strings")
        return tuple(value)

    def _condition(self, row: Mapping[str, Any], condition: Mapping[str, Any]) -> bool:
        field = self._required_string(condition, "field")
        operator = str(condition.get("op", "eq")).casefold()
        actual = _get(row, field)
        expected = condition.get("value")
        if operator == "is_null":
            return actual in (MISSING, None)
        if operator == "not_null":
            return actual not in (MISSING, None)
        if actual is MISSING:
            return False
        operations = {
            "eq": lambda: actual == expected,
            "ne": lambda: actual != expected,
            "lt": lambda: actual < expected,
            "lte": lambda: actual <= expected,
            "gt": lambda: actual > expected,
            "gte": lambda: actual >= expected,
            "in": lambda: actual in expected,
            "not_in": lambda: actual not in expected,
            "contains": lambda: expected in actual,
            "starts_with": lambda: str(actual).startswith(str(expected)),
            "ends_with": lambda: str(actual).endswith(str(expected)),
            "before": lambda: (_parse_time(actual) is not None and _parse_time(expected) is not None and _parse_time(actual) < _parse_time(expected)),
            "after": lambda: (_parse_time(actual) is not None and _parse_time(expected) is not None and _parse_time(actual) > _parse_time(expected)),
        }
        if operator not in operations:
            raise ValidationError(f"unsupported FILTER operator: {operator}")
        try:
            return bool(operations[operator]())
        except (TypeError, ValueError):
            return False

    def _join(
        self,
        left: list[dict[str, Any]],
        right: list[dict[str, Any]],
        params: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        on = params.get("on")
        if (
            isinstance(on, list)
            and bool(on)
            and all(isinstance(item, str) and bool(item) for item in on)
            and len(on) == len(set(on))
        ):
            mapping = {item: item for item in on}
        elif (
            isinstance(on, Mapping)
            and bool(on)
            and all(
                isinstance(left_field, str)
                and bool(left_field)
                and isinstance(right_field, str)
                and bool(right_field)
                for left_field, right_field in on.items()
            )
        ):
            mapping = dict(on)
        else:
            raise ValidationError(
                "JOIN on must be a non-empty, duplicate-free field list or "
                "non-empty left-to-right field mapping"
            )
        how = str(params.get("how", "inner")).casefold()
        if how not in {"inner", "left"}:
            raise ValidationError("JOIN supports inner and left modes")
        right_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in right:
            key = self._join_key(row, mapping.values())
            if key is not None:
                right_index[key].append(row)
        output: list[dict[str, Any]] = []
        for left_row in left:
            lookup = self._join_key(left_row, mapping)
            matches = [] if lookup is None else right_index.get(lookup, [])
            if not matches and how == "left":
                output.append(deepcopy(left_row))
            for right_row in matches:
                joined = deepcopy(left_row)
                for key, value in right_row.items():
                    if key == "__witness_ids":
                        continue
                    destination = key
                    while destination in joined:
                        destination = f"right.{destination}"
                    joined[destination] = deepcopy(value)
                joined["__witness_ids"] = _witness_ids([left_row, right_row])
                output.append(joined)
        return output

    @staticmethod
    def _join_key(row: Mapping[str, Any], fields: Iterable[str]) -> str | None:
        """Return a typed equijoin key, or ``None`` for an unknown key.

        JOIN follows SQL's default null semantics: a row with a missing or null
        value in any key component cannot match another row.  In particular,
        missing fields are not normalized to JSON null and allowed to match one
        another.  Canonical JSON deliberately preserves the remaining value
        types rather than applying implicit string/numeric coercions.
        """

        values: list[Any] = []
        for field in fields:
            value = _get(row, field)
            if value is MISSING or value is None:
                return None
            values.append(value)
        return canonical_json(values)

    def _group_by(self, rows: list[dict[str, Any]], params: Mapping[str, Any]) -> list[dict[str, Any]]:
        fields = self._fields(params)
        aggregates = params.get("aggregates", {"count": {"op": "COUNT"}})
        if not isinstance(aggregates, Mapping):
            raise ValidationError("GROUP_BY aggregates must be an object")
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        keys: dict[str, tuple[Any, ...]] = {}
        for row in rows:
            values = _key(row, fields)
            encoded = canonical_json(values)
            keys[encoded] = values
            groups[encoded].append(row)
        output: list[dict[str, Any]] = []
        for encoded in sorted(groups):
            members = groups[encoded]
            item = dict(zip(fields, keys[encoded]))
            for output_name, aggregate in aggregates.items():
                if not isinstance(aggregate, Mapping):
                    raise ValidationError("aggregate definitions must be objects")
                operation = str(aggregate.get("op", "COUNT")).upper()
                field = aggregate.get("field")
                values = [] if field is None else [
                    value for row in members if (value := _get(row, str(field))) not in (MISSING, None)
                ]
                if operation == "COUNT":
                    item[str(output_name)] = len(members) if field is None else len(values)
                elif operation == "SUM":
                    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
                        raise ValidationError("GROUP_BY SUM accepts only numeric values")
                    item[str(output_name)] = sum(values)
                elif operation == "MIN":
                    item[str(output_name)] = min(values) if values else None
                elif operation == "MAX":
                    item[str(output_name)] = max(values) if values else None
                else:
                    raise ValidationError(f"unsupported GROUP_BY aggregate: {operation}")
            item["__witness_ids"] = _witness_ids(members)
            output.append(item)
        return output

    def _temporal_resolve(
        self, rows: list[dict[str, Any]], params: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        group_fields = self._fields(params, required=False)
        valid_field = str(params.get("valid_time_field", "valid_time"))
        valid_to_field = str(params.get("valid_to_field", "valid_to"))
        record_field = str(params.get("record_time_field", "record_time"))
        as_of = _parse_time(params.get("as_of"))
        recorded_as_of = _parse_time(params.get("recorded_as_of"))
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[canonical_json(_key(row, group_fields) if group_fields else ["__all__"])].append(row)
        result: list[dict[str, Any]] = []
        for encoded in sorted(groups):
            eligible: list[tuple[datetime, datetime, str, dict[str, Any]]] = []
            for row in groups[encoded]:
                valid = _parse_time(_get(row, valid_field, None)) or datetime.min
                if valid.tzinfo is None:
                    valid = valid.replace(tzinfo=timezone.utc)
                valid_to = _parse_time(_get(row, valid_to_field, None))
                recorded = _parse_time(_get(row, record_field, None)) or datetime.min.replace(
                    tzinfo=timezone.utc
                )
                if as_of is not None and valid > as_of:
                    continue
                if as_of is not None and valid_to is not None and as_of >= valid_to:
                    continue
                if recorded_as_of is not None and recorded > recorded_as_of:
                    continue
                eligible.append((valid, recorded, canonical_json(row), row))
            if eligible:
                selected = max(eligible, key=lambda item: (item[0], item[1], item[2]))[3]
                result.append(deepcopy(selected))
        return result

    def _set_operation(
        self,
        operation: str,
        inputs: list[list[dict[str, Any]]],
        params: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        fields = self._fields(params, required=False)

        def key_for(row: Mapping[str, Any]) -> str:
            if fields:
                return canonical_json(_key(row, fields))
            return canonical_json({key: value for key, value in row.items() if not key.startswith("__")})

        maps: list[dict[str, list[dict[str, Any]]]] = []
        for relation in inputs:
            mapping: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in relation:
                mapping[key_for(row)].append(row)
            maps.append(mapping)
        if operation == "UNION":
            selected_keys = set().union(*(mapping for mapping in maps))
        elif operation == "INTERSECTION":
            selected_keys = set(maps[0]).intersection(*(mapping for mapping in maps[1:]))
        else:
            selected_keys = set(maps[0]).difference(*(mapping for mapping in maps[1:]))
        output: list[dict[str, Any]] = []
        for key in sorted(selected_keys):
            candidates = [row for mapping in maps for row in mapping.get(key, [])]
            item = deepcopy(candidates[0])
            item["__witness_ids"] = _witness_ids(candidates)
            output.append(item)
        return output
