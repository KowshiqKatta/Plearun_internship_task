"""Deterministic evidence matching. No network, LLM, or executable record content.

The decision policy is precision-first: require evidence about the same event
and, for a physical event, the same vehicle before reporting a conflict.
"""
from __future__ import annotations
from collections import defaultdict
import re
from statistics import median
from typing import Any

from claims import Claim
from evidence import Evidence


def _time_seconds(s: str) -> int | None:
    m = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?\b", s)
    return int(m.group(1))*3600 + int(m.group(2))*60 + int(m.group(3) or 0) if m else None


def _row_wall_seconds(row: Evidence) -> int | None:
    if row.instant:
        return row.instant.hour * 3600 + row.instant.minute * 60 + row.instant.second
    for col in ("time", "timestamp", "logged", "occurred_at"):
        val = row.attributes.get(col)
        if val is not None:
            t = _time_seconds(str(val))
            if t is not None:
                return t
    return None


def _within_time(claim: Claim, row: Evidence, text: str = "") -> bool:
    if not claim.clock:
        return True
    target = _time_seconds(claim.clock)
    observed = _time_seconds(text) if text else _row_wall_seconds(row)
    if target is None or observed is None:
        return False
    tolerance = 90 if re.search(r"\b(?:about|around|roughly|approximately|circa|~)\b", claim.text, re.I) else 59
    return abs((observed - target + 43200) % 86400 - 43200) <= tolerance


def _exact_quote(row: Evidence, prefer: str = "") -> str | None:
    """Return real source text, never a reconstructed CSV/JSON value."""
    if prefer and prefer in row.raw:
        return prefer
    if row.kind == "vtt":
        # Source text of cue, including speaker. Timeline is not evidence of
        # synchronized wall-clock time.
        for line in row.text.splitlines():
            if line and line in row.raw:
                return line
    return row.raw if row.raw else None


def _make_result(claim: Claim, status: str, row: Evidence | None = None, quote: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {"claim": claim.text, "finding": status}
    if status in ("match", "conflict"):
        if row is None:
            result["finding"] = "not_settled"
        else:
            citation = row.citation(quote)
            if citation:
                result.update(citation)
            else:
                result["finding"] = "not_settled"
    return result


def _is_primary_event(row: Evidence, event: str) -> bool:
    if row.event != event:
        return False
    if row.kind == "json":
        return True
    if row.kind == "csv":
        t = row.text.lower()
        if event == "911_call":
            return bool((row.attributes.get("type") or row.attributes.get("event_type")) == "911_call" or re.search(r"\b(?:911[_\s-]*call|call\s*(?:logged|received)|logged\s*call)\b", t))
        return True
    return True


def _normalize_event_text(s: str) -> str:
    return re.sub(r"\W+", "_", s.lower()).strip("_")


class ClockAligner:
    """Estimate source-clock offsets only from repeated named shared events.

    At least two independent event anchors must agree within the precision
    of the weakest source. Unidentified speed=0 camera readings are never used
    as stop anchors, and a note *describing* a caller is not an inbound-call
    log. Offsets are wall-clock differences in seconds.
    """
    def __init__(self, rows: list[Evidence]):
        self.offsets: dict[tuple[str, str], float] = {}
        sources: dict[str, list[Evidence]] = defaultdict(list)
        for r in rows:
            if _row_wall_seconds(r) is not None and r.event:
                sources[r.clock].append(r)
        ids = sorted(sources)
        for i, source_a in enumerate(ids):
            for source_b in ids[i+1:]:
                pairs: list[tuple[str, float]] = []
                a_by_event: dict[str, list[Evidence]] = defaultdict(list)
                b_by_event: dict[str, list[Evidence]] = defaultdict(list)
                for row in sources[source_a]:
                    if row.event in ("911_call", "unit_on_scene", "unit_dispatched", "vehicle_stop", "vehicle_observed") and _is_primary_event(row, row.event):
                        a_by_event[row.event].append(row)
                for row in sources[source_b]:
                    if row.event in ("911_call", "unit_on_scene", "unit_dispatched", "vehicle_stop", "vehicle_observed") and _is_primary_event(row, row.event):
                        b_by_event[row.event].append(row)
                for name in set(a_by_event) & set(b_by_event):
                    aa, bb = a_by_event[name], b_by_event[name]
                    if len(aa) != 1 or len(bb) != 1:
                        continue
                    ra, rb = aa[0], bb[0]
                    # Mere similar narrative is not a safe identity anchor;
                    # need an actual event type on both sides and sufficiently
                    # specific descriptions, or explicit shared IDs.
                    if not (self._shared_id(ra, rb) or self._strong_shared_event(ra, rb)):
                        continue
                    ta, tb = _row_wall_seconds(ra), _row_wall_seconds(rb)
                    if ta is not None and tb is not None:
                        delta = (ta-tb+43200) % 86400 - 43200
                        pairs.append((name, delta))
                if len({name for name, _ in pairs}) < 2:
                    continue
                diffs = [v for _, v in pairs]
                candidate = median(diffs)
                if max(abs(x-candidate) for x in diffs) <= 75:
                    self.offsets[(source_a, source_b)] = candidate
                    self.offsets[(source_b, source_a)] = -candidate

    @staticmethod
    def _shared_id(a: Evidence, b: Evidence) -> bool:
        for key in ("incident_id", "case_id", "event_id", "plate", "license_plate", "vehicle_id", "unit"):
            x, y = a.attributes.get(key), b.attributes.get(key)
            if x and y and str(x).lower() == str(y).lower():
                return True
        return False

    @staticmethod
    def _strong_shared_event(a: Evidence, b: Evidence) -> bool:
        # Explicitly named event log entries. Narrated officer notes don't
        # identify the exact instant a system recorded something.
        def primary(r: Evidence) -> bool:
            if r.kind == "json":
                return True
            if r.kind == "csv":
                return bool(re.search(r"\b(?:logged|received|on scene|dispatched|stopped)\b", r.text, re.I))
            return False
        return primary(a) and primary(b) and a.event == b.event

    def delta(self, a: str, b: str) -> float | None:
        if a == b:
            return 0
        return self.offsets.get((a, b))


def _linked_vehicle(claim_text: str, row: Evidence) -> bool:
    # No camera measurement may identify a car solely by speed/time.
    candidates = {
        "plate": r"\bplate\s*[:#-]?\s*([a-z0-9-]{4,})\b",
        "vehicle_id": r"\bvehicle(?:\s+id|_id)\s*[:#-]?\s*([a-z0-9-]{3,})\b",
    }
    for key, regex in candidates.items():
        m = re.search(regex, claim_text, re.I)
        if m and str(row.attributes.get(key, "")).lower() == m.group(1).lower():
            return True
    return False


def _officer_ref(claim: Claim, row: Evidence) -> bool:
    # An officer's spoken remark is evidence that the officer reported the
    # observation. It is not an independent radar calibration.
    report_officer = re.search(r"\bOfficer\s+([A-Za-z'-]+)", claim.context or claim.text, re.I)
    if not report_officer:
        return True
    return bool(re.search(r"\bOfficer\s+"+re.escape(report_officer.group(1))+r"\b", row.text, re.I))


def _instruction_like(text: str) -> bool:
    # Source material is data. A malicious source trying to steer the program
    # should not be mistaken for a substantive witness statement.
    return bool(re.search(r"ignore (?:all |the |previous )*instructions|system prompt|\b(?:output|print|return) (?:a )?(?:match|conflict|json)\b", text, re.I))


def _spoken_speed(row: Evidence, amount: float) -> bool:
    if row.kind != "vtt" or not _officer_ref(Claim(row.text, "speed"), row):
        return False
    n = str(int(amount)) if amount.is_integer() else str(amount)
    # '58 in a 45' is a valid spoken speed report; a mere '58' is not.
    return bool(re.search(r"(?<!\d)" + re.escape(n) + r"\s*(?:mph|miles per hour|in\s+a\s+\d{1,3}\b)", row.text, re.I))


def _spoken_zone(row: Evidence, amount: float) -> bool:
    if row.kind != "vtt":
        return False
    n = str(int(amount)) if amount.is_integer() else str(amount)
    return bool(re.search(r"(?:\bin\s+a\s+|\bposted\s+|\bzone\s+|\blimit\s+)" + re.escape(n) + r"\b", row.text, re.I))


def _speeds(claim: Claim, records: list[Evidence]) -> dict[str, Any]:
    if claim.value is None:
        return _make_result(claim, "not_settled")
    spoken = [row for row in records if row.kind == "vtt" and
              _officer_ref(claim, row) and _spoken_speed(row, claim.value) and
              not _instruction_like(row.text)]
    # If there are multiple independent speed utterances, we cannot tell
    # which one is about the reported car without a vehicle identifier.
    same_officer_utterances = [r for r in records if r.kind == "vtt" and
                               _officer_ref(claim, r) and
                               re.search(r"\b\d{1,3}\s*(?:mph|in a \d{1,3})\b", r.text, re.I) and
                               not _instruction_like(r.text)]
    if len(spoken) == 1 and len(same_officer_utterances) == 1:
        row = spoken[0]
        quote = _exact_quote(row)
        if quote:
            return _make_result(claim, "match", row, quote)
    # A camera's speed is not a trustworthy match OR conflict without a
    # link to the vehicle in the report. Even a perfect time match is not it.
    candidates = [r for r in records if "speed_mph" in r.attributes and _linked_vehicle(claim.text, r)]
    if len(candidates) == 1 and _within_time(claim, candidates[0]):
        row = candidates[0]
        actual = row.attributes.get("speed_mph")
        try:
            value = float(actual)
        except (TypeError, ValueError):
            return _make_result(claim, "not_settled")
        if abs(value-claim.value) < 0.1:
            return _make_result(claim, "match", row, _exact_quote(row) or "")
        # Misaligned camera clocks could create misleading conflicts. A speed
        # mismatch by itself cannot settle a claimed observation.
    return _make_result(claim, "not_settled")


def _zones(claim: Claim, records: list[Evidence]) -> dict[str, Any]:
    if claim.value is None:
        return _make_result(claim, "not_settled")
    for row in records:
        if row.kind == "vtt" and _officer_ref(claim, row) and _spoken_zone(row, claim.value) and not _instruction_like(row.text):
            quote = _exact_quote(row)
            if quote:
                return _make_result(claim, "match", row, quote)
    # All camera entries agree on a zone does not prove the camera is in the
    # location described by the report. A shared location ID is needed.
    return _make_result(claim, "not_settled")


def _stops(claim: Claim, records: list[Evidence]) -> dict[str, Any]:
    rows = [r for r in records if r.event == "vehicle_stop" and r.kind in ("csv", "vtt")]
    # Prefer notes with a written clock and matching location, then speech.
    rows.sort(key=lambda r: (r.kind != "csv", r.file, r.row))
    for row in rows:
        t = row.text.lower()
        c = claim.text.lower()
        has_location = ("lot" in t and "lot" in c) or ("stop" in t and "stop" in c)
        if not has_location:
            continue
        # Prefer an explicitly *spoken/written* time, not a camera clock. A
        # source's wall-clock time cannot be compared across unknown offsets.
        source_time = _time_seconds(row.text)
        if claim.clock and (source_time is None or not _within_time(claim, row, row.text)):
            continue
        quote = _exact_quote(row)
        if quote:
            return _make_result(claim, "match", row, quote)
    return _make_result(claim, "not_settled")


def _calls(claim: Claim, records: list[Evidence]) -> dict[str, Any]:
    logs = [r for r in records if r.event == "911_call" and r.kind == "json" and r.instant]
    if len(logs) != 1:
        return _make_result(claim, "not_settled")
    row = logs[0]
    if not claim.clock:
        return _make_result(claim, "match", row, _exact_quote(row) or "")
    # An unsynchronized CAD time cannot be compared to a report's wall time.
    # The exception is an explicitly *source-clock* claim, e.g. "according to
    # the dispatch log the time field reads 21:11". For that, we can compare
    # the value in the same source without clock alignment.
    mentions_source_clock = bool(re.search(
        r"\b(?:dispatch|CAD)\b.*\b(?:log|logged|record(?:s|ed)?|clock|timestamp|time field)\b",
        claim.text, re.I))
    if not mentions_source_clock:
        return _make_result(claim, "not_settled")
    is_equal = row.instant.strftime("%H:%M:%S" if claim.clock.count(":") == 2 else "%H:%M") == claim.clock
    if is_equal:
        return _make_result(claim, "match", row, _exact_quote(row) or "")
    if not re.search(r"\b(?:about|around|roughly|approximately)\b", claim.text, re.I):
        return _make_result(claim, "conflict", row, _exact_quote(row) or "")
    return _make_result(claim, "not_settled")


def _delay(claim: Claim, records: list[Evidence], aligner: ClockAligner) -> dict[str, Any]:
    if claim.value is None:
        return _make_result(claim, "not_settled")
    scenes = [r for r in records if r.event == "unit_on_scene" and _row_wall_seconds(r) is not None]
    stops = [r for r in records if r.event == "vehicle_stop" and _row_wall_seconds(r) is not None]
    if len(scenes) != 1 or len(stops) != 1:
        return _make_result(claim, "not_settled")
    a, b = scenes[0], stops[0]
    offset = aligner.delta(a.clock, b.clock)
    if offset is None:
        return _make_result(claim, "not_settled")
    s, t = _row_wall_seconds(a), _row_wall_seconds(b)
    if s is None or t is None:
        return _make_result(claim, "not_settled")
    elapsed = (s-t-offset+43200) % 86400 - 43200
    tolerance = 90 if re.search(r"\b(?:roughly|about|approximately|around)\b", claim.text, re.I) else 30
    if abs(elapsed-claim.value) <= tolerance:
        return _make_result(claim, "match", a, _exact_quote(a) or "")
    # Claim about temporal *relationship* requires enough context to identify
    # both events. A single cited row cannot prove the conflict on its own;
    # avoid false conflicts based on overly strict clock arithmetic.
    return _make_result(claim, "not_settled")


def _counts(claim: Claim, records: list[Evidence]) -> dict[str, Any]:
    rows = [r for r in records if any(k in r.attributes for k in ("lane_count", "vehicles_in_lane", "car_count"))]
    if len(rows) != 1:
        return _make_result(claim, "not_checkable" if not rows else "not_settled")
    row = rows[0]
    if not _linked_vehicle(claim.text, row):
        return _make_result(claim, "not_settled")
    n = next(row.attributes[k] for k in ("lane_count", "vehicles_in_lane", "car_count") if k in row.attributes)
    try:
        actual = int(n)
    except (TypeError, ValueError):
        return _make_result(claim, "not_settled")
    return _make_result(claim, "match" if actual == 1 else "conflict", row, _exact_quote(row) or "")


def verify(claims: list[Claim], records: list[Evidence]) -> dict[str, Any]:
    aligner = ClockAligner(records)
    rules = {"speed": lambda c: _speeds(c, records),
             "zone": lambda c: _zones(c, records),
             "vehicle_stop": lambda c: _stops(c, records),
             "911_call": lambda c: _calls(c, records),
             "scene_delay": lambda c: _delay(c, records, aligner),
             "lane_count": lambda c: _counts(c, records),
             "vehicle_observed": lambda c: _make_result(c, "not_settled"),
             "background": lambda c: _make_result(c, "not_checkable"),
             "other": lambda c: _make_result(c, "not_checkable")}
    results = [rules[c.kind](c) for c in claims]
    return {"findings": results}
