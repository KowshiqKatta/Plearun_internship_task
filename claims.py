"""Conservative, source-exact extraction of checkable report propositions."""
from __future__ import annotations
from dataclasses import dataclass, replace
import re

TIME_RX = re.compile(r"\b(?:(?:at|about|around|roughly|approximately)\s+)?(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?\s*(?:a\.?m\.?|p\.?m\.?)?\b", re.I)
SPEED_RX = re.compile(r"\b(\d{1,3}(?:\.\d+)?)\s*(?:mph|mi/h|miles per hour)\b", re.I)
ZONE_RX = re.compile(r"\b(?:posted|speed\s+limit(?:\s+of)?|zone(?:\s+(?:of|is))?)\s*(\d{1,3})\s*(?:mph)?\s*(?:zone)?\b", re.I)
DATE_RX = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


@dataclass(frozen=True)
class Claim:
    text: str
    kind: str
    value: float | None = None
    clock: str = ""
    context: str = ""


def classify(text: str) -> Claim:
    lower = text.lower()
    times = TIME_RX.findall(text)
    clock = re.search(r"\b([0-2]?\d:[0-5]\d(?::[0-5]\d)?)\b", text)
    speed = SPEED_RX.search(text)
    zone = ZONE_RX.search(text)
    if re.search(r"\b(?:only|sole|no other|single|just one)\b.*\b(?:car|vehicle|lane|automobile)\b", lower):
        return Claim(text, "lane_count", 1, clock.group(1) if clock else "")
    if "911" in lower and re.search(r"\b(?:call|logged|received|reported|dialled|dialed)\b", lower):
        return Claim(text, "911_call", None, clock.group(1) if clock else "")
    if re.search(r"\b(?:unit|officer|patrol)\b.*\b(?:on.scene|arriv(?:ed|al))\b", lower) and re.search(r"\b(?:minutes?|mins?|seconds?|secs?)\b", lower):
        mins = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:minutes?|mins?)\b", lower)
        secs = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:seconds?|secs?)\b", lower)
        return Claim(text, "scene_delay", float(mins.group(1))*60 if mins else float(secs.group(1)) if secs else None)
    if re.search(r"\b(?:stopped|pulled into|pulling in|pulled over|came to a stop)\b", lower):
        return Claim(text, "vehicle_stop", None, clock.group(1) if clock else "")
    if speed:
        return Claim(text, "speed", float(speed.group(1)), clock.group(1) if clock else "")
    if zone:
        return Claim(text, "zone", float(zone.group(1)), clock.group(1) if clock else "")
    if re.search(r"\b(?:first observed|spotted|first saw|observed|sighted)\b", lower):
        return Claim(text, "vehicle_observed", None, clock.group(1) if clock else "")
    if DATE_RX.search(text) or re.search(r"\b(?:route|mile marker)\s+\d+", lower):
        return Claim(text, "background")
    return Claim(text, "other")


def extract_claims(report: str) -> list[Claim]:
    # Preserve exact source substrings for the 'claim' output field.
    source = re.sub(r"(?m)^\s*SYNTHETIC\s*-\s*FICTIONAL CASE\.?\s*", "", report, flags=re.I)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", source.strip())
    fragments: list[str] = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        # Separate the independently testable numeric assertions, while
        # keeping exact contiguous spans from the report.
        parts = re.split(r",\s+(?=(?:travell?ing|moving|driving)\b)", sent, flags=re.I)
        expanded: list[str] = []
        for part in parts:
            if SPEED_RX.search(part) and re.search(r"\b(?:in a posted|in a \d+ zone|speed limit)\b", part, re.I):
                split = re.split(r"\s+(?=in a posted\s+\d+\b|in a \d+\s+zone\b)", part, maxsplit=1, flags=re.I)
                expanded.extend(split)
            else:
                expanded.append(part)
        fragments.extend(expanded)
    claims: list[Claim] = []
    seen: set[str] = set()
    for frag in fragments:
        frag = frag.strip()
        if frag and frag not in seen:
            seen.add(frag)
            claims.append(replace(classify(frag), context=next((s for s in sentences if frag in s), frag)))
    return claims
