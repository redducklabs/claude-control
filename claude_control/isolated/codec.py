"""Strict, bounded JSON decoding for isolated-runner protocol objects."""

from __future__ import annotations

import math
import re
from typing import NoReturn, cast

from .contracts import JsonValue, Refusal

_MAX_DEPTH = 32
_MAX_KEYS = 256
_MAX_ARRAY_ELEMENTS = 4096
_MAX_STRING_BYTES = 262144
_MAX_NUMBER_CHARS = 128
_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_DERIVED_FIELDS = frozenset(
    {
        "job_id",
        "repository_id",
        "resolved_repository_id",
        "repository_identity",
        "resolved_sha",
        "resolved_head",
        "resolved_base",
        "policy_version",
        "policy_digest",
        "qualification_digest",
        "provenance",
        "authorship_provenance",
        "authoring_set",
        "model_identity",
        "resolved_model_identity",
        "model_pin",
        "model_version",
        "topology_identity",
        "resolved_topology_identity",
        "topology_fingerprint",
    }
)


def _refuse(reason: str) -> NoReturn:
    raise Refusal("invalid transport", reason)


class _Parser:
    """Recursive descent parser whose bounds are checked before container growth."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.index = 0

    def parse(self) -> JsonValue:
        self._space()
        value = self._value(0)
        self._space()
        if self.index != len(self.text):
            _refuse("trailing data after JSON document")
        return value

    def _space(self) -> None:
        text = self.text
        length = len(text)
        while self.index < length and text[self.index] in " \t\r\n":
            self.index += 1

    def _value(self, depth: int) -> JsonValue:
        if self.index >= len(self.text):
            _refuse("truncated JSON document")
        char = self.text[self.index]
        if char == "{":
            return self._object(depth + 1)
        if char == "[":
            return self._array(depth + 1)
        if char == '"':
            return self._string()
        for literal, value in (("true", True), ("false", False), ("null", None)):
            if self.text.startswith(literal, self.index):
                self.index += len(literal)
                return value
        return self._number()

    def _object(self, depth: int) -> dict[str, JsonValue]:
        if depth > _MAX_DEPTH:
            _refuse("JSON nesting depth exceeded")
        self.index += 1
        self._space()
        result: dict[str, JsonValue] = {}
        if self._consume("}"):
            return result
        while True:
            if len(result) >= _MAX_KEYS:
                _refuse("JSON object key count exceeded")
            if self.index >= len(self.text) or self.text[self.index] != '"':
                _refuse("object key must be a string")
            key = self._string()
            if key in result:
                _refuse("duplicate JSON object key")
            self._space()
            if not self._consume(":"):
                _refuse("object key missing colon")
            self._space()
            result[key] = self._value(depth)
            self._space()
            if self._consume("}"):
                return result
            if not self._consume(","):
                _refuse("object member separator expected")
            self._space()

    def _array(self, depth: int) -> list[JsonValue]:
        if depth > _MAX_DEPTH:
            _refuse("JSON nesting depth exceeded")
        self.index += 1
        self._space()
        result: list[JsonValue] = []
        if self._consume("]"):
            return result
        while True:
            if len(result) >= _MAX_ARRAY_ELEMENTS:
                _refuse("JSON array element count exceeded")
            result.append(self._value(depth))
            self._space()
            if self._consume("]"):
                return result
            if not self._consume(","):
                _refuse("array element separator expected")
            self._space()

    def _string(self) -> str:
        start = self.index
        self.index += 1
        escaped = False
        while self.index < len(self.text):
            char = self.text[self.index]
            self.index += 1
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                raw = self.text[start : self.index]
                try:
                    import json

                    value = cast(str, json.loads(raw))
                except (ValueError, TypeError) as exc:
                    raise Refusal("invalid transport", "invalid JSON string") from exc
                try:
                    decoded_bytes = value.encode("utf-8", errors="strict")
                except UnicodeEncodeError as exc:
                    raise Refusal(
                        "invalid transport", "JSON string contains an invalid surrogate"
                    ) from exc
                if len(decoded_bytes) > _MAX_STRING_BYTES:
                    _refuse("JSON string byte limit exceeded")
                return value
            elif ord(char) < 0x20:
                _refuse("unescaped control character in JSON string")
        _refuse("truncated JSON string")
        raise AssertionError("unreachable")

    def _number(self) -> int | float:
        match = _NUMBER.match(self.text, self.index)
        if match is None:
            _refuse("invalid JSON value")
        token = match.group()
        if len(token) > _MAX_NUMBER_CHARS:
            _refuse("JSON number token limit exceeded")
        self.index = match.end()
        try:
            value: int | float = (
                float(token) if any(c in token for c in ".eE") else int(token)
            )
        except (ValueError, OverflowError) as exc:
            raise Refusal("invalid transport", "invalid JSON number") from exc
        if type(value) is float and not math.isfinite(value):
            _refuse("JSON number must be finite")
        return value

    def _consume(self, char: str) -> bool:
        if self.index < len(self.text) and self.text[self.index] == char:
            self.index += 1
            return True
        return False


def _contains_derived_field(value: JsonValue) -> bool:
    if isinstance(value, dict):
        return any(
            key in _DERIVED_FIELDS or _contains_derived_field(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_derived_field(item) for item in value)
    return False


def _decode_object(
    raw: bytes, max_bytes: int, *, reject_derived: bool
) -> dict[str, JsonValue]:
    if type(max_bytes) is not int or max_bytes <= 0:
        _refuse("invalid byte limit")
    if max_bytes > (1 << 20):
        _refuse("document byte limit exceeds the published event limit")
    if not isinstance(raw, bytes):
        _refuse("protocol input must be bytes")
    if len(raw) > max_bytes:
        _refuse("JSON document byte limit exceeded")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise Refusal("invalid transport", "protocol is not valid UTF-8") from exc
    value = _Parser(text).parse()
    if not isinstance(value, dict):
        _refuse("protocol document must be an object")
    if reject_derived and _contains_derived_field(value):
        _refuse("inbound request contains a runner-derived field")
    return value


def decode_object(raw: bytes, max_bytes: int) -> dict[str, JsonValue]:
    """Decode one inbound UTF-8 JSON object and refuse runner-derived fields."""
    return _decode_object(raw, max_bytes, reject_derived=True)
