"""Lossless corpus storage and stable character-offset partitioning."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import os
from typing import Any, Iterable, Iterator, Mapping

from .models import Document, JSONValue, SourceBlock, ValidationError, content_hash, stable_hash


def _value(source: object, *names: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
        return default
    for name in names:
        if hasattr(source, name):
            return getattr(source, name)
    return default


def coerce_document(source: Document | Mapping[str, Any] | object) -> Document:
    """Coerce common SourceDocument-like shapes without discarding raw text."""

    if isinstance(source, Document):
        return source
    document_id = _value(source, "document_id", "doc_id", "source_id", "id")
    raw_content = _value(source, "raw_content", "content", "text", "body")
    if document_id is None or raw_content is None:
        raise ValidationError("document input needs an id and raw content/text")
    metadata = _value(source, "metadata", default={}) or {}
    if not isinstance(metadata, Mapping):
        metadata = {"adapter_metadata": str(metadata)}
    structure = _value(source, "structure", default={}) or {}
    if not isinstance(structure, Mapping):
        structure = {"adapter_structure": str(structure)}
    # Metadata remains available without silently overriding explicit structure.
    combined_structure: dict[str, JSONValue] = dict(structure)
    for key, value in metadata.items():
        if key not in combined_structure:
            try:
                json.dumps(value, allow_nan=False)
                combined_structure[str(key)] = value
            except (TypeError, ValueError):
                combined_structure[str(key)] = str(value)
    supplied_hash = str(_value(source, "content_hash", default="") or "")
    if supplied_hash.startswith("sha256:"):
        supplied_hash = supplied_hash.removeprefix("sha256:")
    return Document(
        document_id=str(document_id),
        document_version=str(_value(source, "document_version", "version", default="1")),
        source_type=str(_value(source, "source_type", "type", default="text")),
        raw_content=str(raw_content),
        structure=combined_structure,
        valid_from=_value(source, "valid_from", "valid_time"),
        valid_to=_value(source, "valid_to"),
        record_time=_value(source, "record_time", "timestamp"),
        permissions=(
            (str(_value(source, "permissions", "acl")),)
            if isinstance(_value(source, "permissions", "acl", default=()), str)
            else tuple(_value(source, "permissions", "acl", default=()) or ())
        ),
        content_hash=supplied_hash,
        supersedes_version=_value(source, "supersedes_version"),
    )


def _parse_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValidationError(f"invalid ISO timestamp: {value}") from exc


class CorpusStore:
    """Version-preserving corpus store.

    Documents are immutable by ``(document_id, document_version)``.  Reusing a
    version with different bytes is rejected, preventing offsets and citations
    from silently changing.  If ``storage_path`` is supplied, each version is
    also durably stored as JSON using atomic replacement.
    """

    STORE_VERSION = "corpus-store-0.1.0"

    def __init__(
        self,
        *,
        block_size_chars: int = 12_000,
        overlap_chars: int = 400,
        storage_path: str | Path | None = None,
    ) -> None:
        if block_size_chars < 200:
            raise ValidationError("block_size_chars must be at least 200")
        if overlap_chars < 0 or overlap_chars >= block_size_chars // 2:
            raise ValidationError("overlap_chars must be non-negative and less than half the block size")
        self.block_size_chars = block_size_chars
        self.overlap_chars = overlap_chars
        self.storage_path = Path(storage_path) if storage_path is not None else None
        self._documents: dict[tuple[str, str], Document] = {}
        if self.storage_path is not None:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self._load()

    def _file_name(self, document: Document) -> str:
        digest = stable_hash([document.document_id, document.document_version])[:24]
        return f"{digest}.json"

    def _load(self) -> None:
        assert self.storage_path is not None
        for path in sorted(self.storage_path.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("store_version") != self.STORE_VERSION:
                    raise ValidationError(
                        f"unsupported corpus store version in {path}: {payload.get('store_version')}"
                    )
                document = Document.from_dict(payload["document"])
                self._documents[(document.document_id, document.document_version)] = document
            except (OSError, json.JSONDecodeError, KeyError, ValidationError) as exc:
                raise ValidationError(f"cannot load corpus record {path}: {exc}") from exc

    def _persist(self, document: Document) -> None:
        if self.storage_path is None:
            return
        path = self.storage_path / self._file_name(document)
        temporary = path.with_suffix(".tmp")
        payload = {"store_version": self.STORE_VERSION, "document": document.to_dict()}
        temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)

    def add_document(self, source: Document | Mapping[str, Any] | object) -> Document:
        document = coerce_document(source)
        key = (document.document_id, document.document_version)
        existing = self._documents.get(key)
        if existing is not None:
            if existing.content_hash != document.content_hash or existing.to_dict() != document.to_dict():
                raise ValidationError(
                    f"immutable document version {document.document_id}@{document.document_version} changed"
                )
            return existing
        self._documents[key] = document
        try:
            self._persist(document)
        except Exception:
            self._documents.pop(key, None)
            raise
        return document

    def add_documents(self, sources: Iterable[Document | Mapping[str, Any] | object]) -> tuple[Document, ...]:
        return tuple(self.add_document(source) for source in sources)

    def get_document(self, document_id: str, document_version: str) -> Document:
        try:
            return self._documents[(document_id, document_version)]
        except KeyError as exc:
            raise KeyError(f"unknown document {document_id}@{document_version}") from exc

    def remove_document(self, document_id: str, document_version: str) -> Document:
        """Remove one exact version; cached shard views become unreachable."""

        key = (document_id, document_version)
        try:
            document = self._documents.pop(key)
        except KeyError as exc:
            raise KeyError(f"unknown document {document_id}@{document_version}") from exc
        if self.storage_path is not None:
            path = self.storage_path / self._file_name(document)
            try:
                path.unlink(missing_ok=True)
            except Exception:
                self._documents[key] = document
                raise
        return document

    def documents(
        self,
        *,
        principals: Iterable[str] = (),
        recorded_as_of: str | None = None,
    ) -> tuple[Document, ...]:
        principal_set = set(principals)
        cutoff = _parse_time(recorded_as_of)
        selected: list[Document] = []
        for document in self._documents.values():
            if document.permissions and not principal_set.intersection(document.permissions):
                continue
            record_time = _parse_time(document.record_time)
            if cutoff is not None and record_time is not None and record_time > cutoff:
                continue
            selected.append(document)
        return tuple(sorted(selected, key=lambda item: (item.document_id, item.document_version)))

    def _block_ranges(self, content: str) -> Iterator[tuple[int, int]]:
        if not content:
            yield (0, 0)
            return
        length = len(content)
        start = 0
        while start < length:
            hard_end = min(length, start + self.block_size_chars)
            end = hard_end
            if hard_end < length:
                lower_bound = start + int(self.block_size_chars * 0.75)
                newline = content.rfind("\n", lower_bound, hard_end)
                space = content.rfind(" ", lower_bound, hard_end)
                boundary = max(newline, space)
                if boundary > start:
                    end = boundary + 1
            yield (start, end)
            if end == length:
                break
            next_start = end - self.overlap_chars
            start = max(start + 1, next_start)

    def blocks(
        self,
        *,
        principals: Iterable[str] = (),
        recorded_as_of: str | None = None,
    ) -> tuple[SourceBlock, ...]:
        blocks: list[SourceBlock] = []
        for document in self.documents(principals=principals, recorded_as_of=recorded_as_of):
            for start, end in self._block_ranges(document.raw_content):
                text = document.raw_content[start:end]
                shard_hash = content_hash(text)
                identity = stable_hash(
                    [
                        document.document_id,
                        document.document_version,
                        document.content_hash,
                        start,
                        end,
                        shard_hash,
                    ]
                )[:24]
                blocks.append(
                    SourceBlock(
                        block_id=f"blk-{identity}",
                        document_id=document.document_id,
                        document_version=document.document_version,
                        source_type=document.source_type,
                        start_offset=start,
                        end_offset=end,
                        content=text,
                        document_hash=document.content_hash,
                        block_hash=shard_hash,
                        valid_from=document.valid_from,
                        valid_to=document.valid_to,
                        record_time=document.record_time,
                        permissions=document.permissions,
                        structure=document.structure,
                    )
                )
        return tuple(blocks)

    def snapshot_hash(
        self,
        *,
        principals: Iterable[str] = (),
        recorded_as_of: str | None = None,
    ) -> str:
        documents = self.documents(principals=principals, recorded_as_of=recorded_as_of)
        return "sha256:" + stable_hash(
            {
                "store_version": self.STORE_VERSION,
                "partition": [self.block_size_chars, self.overlap_chars],
                "documents": [
                    {
                        "document_id": item.document_id,
                        "document_version": item.document_version,
                        "content_hash": item.content_hash,
                        "source_type": item.source_type,
                        "structure": dict(item.structure),
                        "valid_from": item.valid_from,
                        "valid_to": item.valid_to,
                        "record_time": item.record_time,
                        "permissions": list(item.permissions),
                        "supersedes_version": item.supersedes_version,
                    }
                    for item in documents
                ],
            }
        )

    def verify_block(self, block: SourceBlock) -> bool:
        try:
            document = self.get_document(block.document_id, block.document_version)
        except KeyError:
            return False
        return (
            document.content_hash == block.document_hash
            and document.raw_content[block.start_offset : block.end_offset] == block.content
            and content_hash(block.content) == block.block_hash
        )

    def __len__(self) -> int:
        return len(self._documents)
