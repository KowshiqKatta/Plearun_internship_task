"""Offline, provenance-preserving readers for mixed incident records.

`row` is a 1-based physical source line for CSV/JSON, or a 1-based cue
number for WebVTT. Quotes are always substrings of source bytes decoded
with the selected encoding (never reformatted extracted cell values).
"""
from __future__ import annotations

import bisect
import csv
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import re
from typing import Any, Iterable

MAX_FILE_BYTES = 100_000_000
TIME_KEYS = ("logged", "timestamp", "time", "occurred_at", "event_time", "datetime", "date_time")
EVENT_KEYS = ("type", "event_type", "action", "event", "status")
SENSITIVE_IDS = ("plate", "license_plate", "vehicle_id", "incident_id", "case_id", "unit")


@dataclass(frozen=True)
class Evidence:
    file: str
    row: int
    raw: str
    text: str
    kind: str
    attributes: dict[str, Any] = field(default_factory=dict)
    clock: str = ""
    instant: datetime | None = None
    event: str = ""

    def quote(self, literal: str) -> str | None:
        """Only allow source-exact quotes. Never cite a synthetic value."""
        if not literal or literal not in self.raw:
            return None
        return literal

    def citation(self, quote: str) -> dict[str, Any] | None:
        if not self.quote(quote):
            return None
        return {"cite": {"file": self.file, "row": self.row}, "quote": quote}


def decode_record(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    raise ValueError("Unable to decode record")


def read_text(path: Path) -> str:
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"Record exceeds {MAX_FILE_BYTES} byte safety limit: {path.name}")
    return decode_record(path.read_bytes())


def timestamp(value: Any, tz: timezone | None = None) -> datetime | None:
    """Read ISO dates, local wall-clock timestamps, and Unix s/ms/us/ns.

    Never assume a timezone for a naive date. A per-file explicit timezone
    may be safely applied to Unix values (which represent instants).
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) or (isinstance(value, str) and re.fullmatch(r"\d{10,19}", value.strip())):
        value = float(value)
        while abs(value) > 100_000_000_000:
            value /= 1000
        try:
            return datetime.fromtimestamp(value, tz or timezone.utc)
        except (ValueError, OverflowError, OSError):
            return None
    if not isinstance(value, str):
        return None
    value = value.strip()
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None and tz is not None:
            return dt.replace(tzinfo=tz)
        return dt
    except ValueError:
        return None


def infer_file_tz(text: str) -> timezone | None:
    # Explicit ISO offsets are a timezone, NOT a device's clock adjustment.
    offsets: set[int] = set()
    for m in re.finditer(r"\b\d{4}-\d\d-\d\d[T ]\d\d:\d\d(?::\d\d(?:\.\d+)?)?([+-])(\d\d):?(\d\d)\b", text):
        secs = (int(m.group(2)) * 60 + int(m.group(3))) * 60
        offsets.add(secs if m.group(1) == "+" else -secs)
        if len(offsets) > 1:
            return None
    return timezone(timedelta(seconds=next(iter(offsets)))) if offsets else None


def event_type(attrs: dict[str, Any], content: str) -> str:
    for key in EVENT_KEYS:
        if key in attrs and attrs[key] is not None:
            s = str(attrs[key]).lower().replace("-", "_").replace(" ", "_")
            if "911" in s or "caller" in s or "emergency_call" in s:
                return "911_call"
            if "on_scene" in s or "arrived" in s or "arrival" in s:
                return "unit_on_scene"
            if "dispatched" in s or "dispatch_sent" in s:
                return "unit_dispatched"
            if "stop" in s or "pull_over" in s:
                return "vehicle_stop"
            if "observ" in s or "sight" in s:
                return "vehicle_observed"
            return s
    t = content.lower()
    if re.search(r"\b(?:stopped|pulling in|pulled into|pull over)\b", t):
        return "vehicle_stop"
    if re.search(r"\b(?:caller|911 call|911_call)\b", t):
        return "911_call"  # reference to call, not necessarily *logged* event
    if "on scene" in t or "arrived" in t:
        return "unit_on_scene"
    if "observ" in t:
        return "vehicle_observed"
    return ""


def _normal_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    return {str(k).strip().lower(): v for k, v in attrs.items()}


def csv_rows(path: Path, rel: str, content: str) -> list[Evidence]:
    lines = content.splitlines(keepends=True)
    if not lines:
        return []
    # Skip record marker / comments, not a data row. BOM already stripped.
    skipped = 0
    while skipped < len(lines) and (not lines[skipped].strip() or lines[skipped].lstrip().startswith("#")):
        skipped += 1
    result: list[Evidence] = []
    reader = csv.reader(io.StringIO("".join(lines[skipped:]), newline=""))
    try:
        header = next(reader)
    except (StopIteration, csv.Error):
        return []
    if not header:
        return []
    header = [h.strip().lstrip("\ufeff").lower() for h in header]
    prev_end = reader.line_num
    tz = infer_file_tz(content)
    for cells in reader:
        start = skipped + prev_end + 1
        end = skipped + reader.line_num
        prev_end = reader.line_num
        if not any(c.strip() for c in cells):
            continue
        raw = "".join(lines[start-1:end]).rstrip("\r\n")
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        attrs = dict(zip(header, cells))
        vals = " ".join(str(v) for v in attrs.values())
        timevalue = next((attrs[k] for k in TIME_KEYS if k in attrs), None)
        # A note 'Time' field can be HH:MM. It's a clock reading, but not a date.
        dt = timestamp(timevalue, tz)
        device = str(attrs.get("device") or attrs.get("source") or path.stem)
        result.append(Evidence(rel, start, raw, vals, "csv", attrs,
                               device, dt, event_type(attrs, vals)))
    return result


def _json_events(obj: Any) -> Iterable[dict[str, Any]]:
    if isinstance(obj, dict):
        has_clock = any(k.lower() in TIME_KEYS for k in obj)
        if has_clock:
            yield obj
        else:
            for value in obj.values():
                yield from _json_events(value)
    elif isinstance(obj, list):
        for x in obj:
            yield from _json_events(x)


def json_rows(path: Path, rel: str, content: str) -> list[Evidence]:
    obj = json.loads(content)
    lines = content.splitlines()
    tz = infer_file_tz(content)
    # Locate source-exact timestamp fields. The logical JSON object and the
    # timestamp field are linked by encounter order in the same parsed source.
    stamp_rx = re.compile(r'"(?:logged|timestamp|time|occurred_at|event_time|datetime|date_time)"\s*:\s*(?:"(?:\\.|[^"\\])*"|-?\d+(?:\.\d+)?)', re.I)
    spans = list(stamp_rx.finditer(content))
    starts = [0] + [m.end() for m in re.finditer(r"\n", content)]
    events = list(_json_events(obj))
    result: list[Evidence] = []
    # JSON decoder preserves dict insertion order, even in nested exports.
    # If a mapping is ambiguous, refuse to fabricate provenance.
    if len(spans) != len(events):
        return []
    for ev, span in zip(events, spans):
        attrs = _normal_attrs(ev)
        val = next((attrs[k] for k in TIME_KEYS if k in attrs), None)
        dt = timestamp(val, tz)
        lineno = bisect.bisect_right(starts, span.start())
        raw = lines[lineno-1] if lineno <= len(lines) else ""
        device = str(attrs.get("device") or attrs.get("system") or path.stem)
        result.append(Evidence(rel, lineno, raw, json.dumps(ev, ensure_ascii=False),
                               "json", attrs, device, dt, event_type(attrs, json.dumps(ev))))
    return result


def vtt_rows(path: Path, rel: str, content: str) -> list[Evidence]:
    # Overlapping cue intervals intentionally remain independent cue IDs.
    rows: list[Evidence] = []
    current: list[str] = []
    cue_number = 0
    def flush() -> None:
        nonlocal cue_number
        if not any("-->" in s for s in current):
            return
        cue_number += 1
        index = next(i for i, s in enumerate(current) if "-->" in s)
        cue_lines = current[index+1:]
        raw = "\n".join(current)
        cue_text = "\n".join(cue_lines)
        rows.append(Evidence(rel, cue_number, raw, cue_text, "vtt", {},
                             path.stem, None, event_type({}, cue_text)))
    for line in content.splitlines() + [""]:
        if not line.strip():
            flush()
            current = []
        else:
            current.append(line)
    return rows


def load_records(folder: Path) -> list[Evidence]:
    if not folder.is_dir():
        raise ValueError(f"Not a records directory: {folder}")
    result: list[Evidence] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if path.suffix.lower() not in (".csv", ".json", ".vtt"):
            continue
        rel = path.relative_to(folder).as_posix()
        try:
            content = read_text(path)
            if path.suffix.lower() == ".csv":
                result.extend(csv_rows(path, rel, content))
            elif path.suffix.lower() == ".json":
                result.extend(json_rows(path, rel, content))
            else:
                result.extend(vtt_rows(path, rel, content))
        except (OSError, ValueError, UnicodeError, csv.Error, json.JSONDecodeError) as exc:
            # A malformed source is not evidence. Continue with valid files.
            import sys
            print(f"Warning: skipped {rel}: {exc}", file=sys.stderr)
    return result
