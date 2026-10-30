"""Deterministic date/time normalisation (LLMs get US daylight-saving offsets wrong).

parse_datetime("July 9, 2024 at 2:00 PM CST") -> ("2024-07-09T14:00:00-05:00", "CST written during daylight
saving time; interpreted as CDT (UTC-5)"). No tzdata needed: US DST rules are computed directly.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_ZONES = {"E": (-5, "Eastern"), "C": (-6, "Central"), "M": (-7, "Mountain"), "P": (-8, "Pacific")}
_TZ = r"\(?\s*([ECMP])([SD]?)T\b"
_TIME = r"(?:\D{0,6}?(\d{1,2}):(\d{2})(?::\d{2})?\s*([AaPp])?\.?[Mm]?\.?)?\s*(?:" + _TZ + ")?"
_PATTERNS = [
    # July 9, 2024 [at 2:00 PM CST]
    (re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})" + _TIME), "mdy_name"),
    # 07/09/2024 [03:00 PM EDT]
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})" + _TIME), "mdy_num"),
    # 27-JUN-2024 14:00:00
    (re.compile(r"\b(\d{1,2})-([A-Za-z]{3})-(\d{4})" + _TIME), "dmy_name"),
]


def _nth_sunday(year: int, month: int, n: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(6 - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def is_us_dst(d: date) -> bool:
    return _nth_sunday(d.year, 3, 2) <= d < _nth_sunday(d.year, 11, 1)


def parse_datetime(text: str) -> tuple[str | None, str | None]:
    """First date(+time+zone) in text -> (ISO string, note). (None, None) when no date is found."""
    for rx, kind in _PATTERNS:
        m = rx.search(text or "")
        if not m:
            continue
        a, b, y, hh, mm, ampm, zone, sd = m.groups()
        try:
            if kind == "mdy_name":
                month, day = _MONTHS[a[:3].lower()], int(b)
            elif kind == "mdy_num":
                month, day = int(a), int(b)
            else:
                month, day = _MONTHS[b[:3].lower()], int(a)
            d = date(int(y), month, day)
        except (KeyError, ValueError):
            continue
        if hh is None:
            return d.isoformat(), None
        hour = int(hh) % 12 + (12 if ampm and ampm.lower() == "p" else 0) if ampm else int(hh)
        local = datetime(d.year, d.month, d.day, hour, int(mm))
        if not zone:
            return local.isoformat(), "no time zone stated"
        base, name = _ZONES[zone]
        dst = is_us_dst(d)
        offset = base + (1 if dst else 0)
        note = None
        if sd == "S" and dst:
            note = f"{zone}ST written during daylight saving time; interpreted as {zone}DT (UTC{offset})"
        elif sd == "D" and not dst:
            note = f"{zone}DT written outside daylight saving time; interpreted as {zone}ST (UTC{offset})"
        return local.replace(tzinfo=timezone(timedelta(hours=offset))).isoformat(), note
    return None, None
