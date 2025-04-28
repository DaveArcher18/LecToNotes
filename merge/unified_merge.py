#!/usr/bin/env python3
"""
unified_merge.py (revised for sample data)
=========================================
Merge **boards.json** and **transcript.json** — formatted as in the sample files just provided — into
a unified **lecture.json**.  Key differences from the original version:

* **Timestamp formats** now accept both ``HH:MM:SS(.mmm)`` **and** ``HH_MM_SS`` (underscores).
* **Transcript records** contain *start* / *end* ranges (rather than a single timestamp).  Spoken
  content is associated with **all** 5‑minute windows that overlap the range.
* The default aggregation window remains **5 minutes = 300 s** (configurable with
  ``--interval``).

Input JSON shapes
-----------------
``boards.json``  :: list of objects with at least
    {"timestamp": "HH_MM_SS", "path": "…", "text": "…"}

``transcript.json``  :: list of objects with fields
    {"start": "HH_MM_SS", "end": "HH_MM_SS", "content": "…"}

Output schema (unchanged)
-------------------------
```
{
  "lecture": "<title>",
  "date": "YYYY-MM-DD",                # if --date supplied
  "segments": [
    {
      "start_time": "HH:MM:SS",
      "end_time":   "HH:MM:SS",
      "spoken_content": "…concatenated speech…",
      "written_content": [ { …board block… }, … ]
    },
    …
  ]
}
```

Usage
-----
```
python unified_merge.py boards.json transcript.json lecture.json \
                      --title "Habiro Cohomology – Lecture 1" --date 2025-04-24
```
All heavy lifting is still done with the standard library only.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

###############################################################################
# Timestamp helpers
###############################################################################

def _normalise_ts(ts: str) -> str:
    """Return *ts* with underscores replaced by colons ("00_03_24" → "00:03:24")."""
    if "_" in ts and ":" not in ts:
        return ts.replace("_", ":")
    return ts


def _parse_timestamp(ts: str) -> float:
    """Convert *ts* (underscores or colons) → seconds since 00:00:00."""
    ts = _normalise_ts(ts)
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S"):
        try:
            t = datetime.strptime(ts, fmt)
            return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1_000_000
        except ValueError:
            continue
    raise ValueError(f"Unrecognised timestamp format: {ts!r}")


def _seconds_to_ts(seconds: float) -> str:
    """seconds → HH:MM:SS[.mmm] string."""
    td = timedelta(seconds=seconds)
    total = int(td.total_seconds())
    millis = int(round((seconds - total) * 1000))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02}:{m:02}:{s:02}{'.%03d' % millis if millis else ''}"

###############################################################################
# Core merge logic
###############################################################################

def merge_sources(
    boards: List[Dict[str, Any]],
    transcript: List[Dict[str, Any]],
    interval_s: int,
    lecture_title: str,
    lecture_date: str | None = None,
) -> Dict[str, Any]:
    """Build the unified lecture structure."""

    # --------------------------------- boards ------------------------------
    for blk in boards:
        blk["_t_sec"] = _parse_timestamp(blk["timestamp"])

    # -------------------------------- transcript ---------------------------
    for seg in transcript:
        start_sec = _parse_timestamp(seg["start"])
        end_sec = _parse_timestamp(seg["end"])
        seg["_start_sec"] = start_sec
        seg["_end_sec"] = end_sec

    if not boards and not transcript:
        raise ValueError("No data found in either input file – nothing to merge.")

    # timeline bounds (include both start & end of spoken ranges)
    t_min = min([
        *(blk["_t_sec"] for blk in boards),
        *(seg["_start_sec"] for seg in transcript)
    ])
    t_max = max([
        *(blk["_t_sec"] for blk in boards),
        *(seg["_end_sec"] for seg in transcript)
    ])

    n_segments = int((t_max - t_min) // interval_s) + 1
    segments: List[Dict[str, Any]] = [{"spoken": [], "written": []} for _ in range(n_segments)]

    # place spoken content
    for seg in transcript:
        idx_start = int((seg["_start_sec"] - t_min) // interval_s)
        idx_end = int((seg["_end_sec"] - t_min) // interval_s)
        for idx in range(idx_start, idx_end + 1):
            if 0 <= idx < n_segments:
                segments[idx]["spoken"].append(seg["content"].strip())

    # place board blocks
    for blk in boards:
        idx = int((blk["_t_sec"] - t_min) // interval_s)
        if 0 <= idx < n_segments:
            segments[idx]["written"].append({k: v for k, v in blk.items() if k != "_t_sec"})

    # build output list, dropping empty segments
    output_segments: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments):
        if not seg["spoken"] and not seg["written"]:
            continue
        start_s = t_min + i * interval_s
        end_s = min(start_s + interval_s, t_max)
        output_segments.append(
            {
                "start_time": _seconds_to_ts(start_s),
                "end_time": _seconds_to_ts(end_s),
                "spoken_content": " ".join(seg["spoken"]).strip(),
                "written_content": seg["written"],
            }
        )

    lecture_json: Dict[str, Any] = {
        "lecture": lecture_title,
        "segments": output_segments,
    }
    if lecture_date:
        lecture_json["date"] = lecture_date

    return lecture_json

###############################################################################
# CLI
###############################################################################

def _cli() -> None:
    p = argparse.ArgumentParser(description="Merge boards.json + transcript.json → lecture.json")
    p.add_argument("-b", "--boards", required=True, type=Path, help="Path to boards.json")
    p.add_argument("-t", "--transcript", required=True, type=Path, help="Path to transcript.json")
    p.add_argument("-o", "--output", type=Path, default=Path("lecture.json"), help="Destination file (default: lecture.json)")
    p.add_argument("--interval", type=int, default=300, help="window length in seconds (default 300)")
    p.add_argument("--title", default="Untitled Lecture")
    p.add_argument("--date", help="YYYY-MM-DD (optional)")
    args = p.parse_args()

    try:
        boards_data = json.loads(args.boards.read_text("utf-8"))
        transcript_data = json.loads(args.transcript.read_text("utf-8"))
    except Exception as exc:
        sys.exit(f"Input parse error: {exc}")

    try:
        merged = merge_sources(
            boards=boards_data,
            transcript=transcript_data,
            interval_s=args.interval,
            lecture_title=args.title,
            lecture_date=args.date,
        )
    except Exception as exc:
        sys.exit(f"Merge failed: {exc}")

    try:
        args.output.write_text(json.dumps(merged, ensure_ascii=False, indent=2), "utf-8")
    except Exception as exc:
        sys.exit(f"Could not write output: {exc}")

    print(f"Wrote {len(merged['segments'])} segments → {args.output}")

if __name__ == "__main__":
    _cli()
