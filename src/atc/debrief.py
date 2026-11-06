"""Post-flight debrief: the session log reviewed by a stronger model, for the
pilot's radio work and for finding controller/simulator problems to fix."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .state import fmt_freq

DEBRIEF_PROMPT = """You are a senior US flight instructor (CFII) and a former FAA air traffic controller.
Review this flight-simulator radio session. The "controller" was an AI system: a small, fast language
model given computed facts (the FACTS blocks) and told to phrase replies per FAA JO 7110.65. Pilot
transmissions were transcribed by speech recognition, so some oddities are transcription errors, not
the pilot's: call those out separately and don't grade the pilot on them.

References to apply: AIM Chapter 4 Section 2 (radio communications phraseology and techniques), the
Pilot/Controller Glossary (P/CG), FAA JO 7110.65 for controller phraseology, 14 CFR 91 where relevant.
Cite the section (e.g. "AIM 4-2-3", "7110.65 3-7-2", "P/CG: 'Roger'") when you can.

Write a Markdown report with these sections:

## Flight summary
Two or three sentences: airports, facilities worked, phases of flight.

## Pilot radio work
- Overall assessment (2-4 sentences) and what was done well.
- A table of the pilot transmissions that need improvement: | Time | Pilot said | Issue | Better phrasing | Reference |
  Look for: initial call format (who you're calling, who you are, where you are, what you want), missing
  ATIS code, incomplete or wrong readbacks (runway assignments, hold short instructions, altitudes, headings,
  frequencies, squawk codes, altimeter), non-standard words ("roger" vs "wilco", "to/for", "repeat"), missing
  callsign on readbacks, wordiness.
- Top three things to practice.

## Controller and simulator review
For every controller reply (or silence) that a real controller would not have said or done, a table:
| Time | Facility | Controller said | Problem | Likely cause | Suggested fix |
Problems include: wrong or non-standard phraseology, wrong facts (runway, wind, altimeter, frequency,
facility name, position), missing required elements (e.g. hold short/crossing instructions, ATIS letter,
altimeter), not responding when it should have (or responding when it should have stayed silent), poor sequencing,
unrealistic service. "Likely cause" is one of: PHRASING (the model worded it badly despite correct facts),
FACTS (the FACTS block itself was wrong or missing something needed), CAPABILITY (the sim doesn't model
this yet, e.g. traffic, IFR clearances), TRANSCRIPTION (it reacted to a misheard pilot call).
"Suggested fix" should be concrete enough to implement (a prompt rule, a fact to compute, a missing feature).

## Transcription problems
Transmissions where speech recognition clearly misheard the pilot, and what was probably said.

Be specific and concise. Don't pad sections; write "None" if a section has nothing.
"""


def _zulu(sec) -> str:
    if sec is None:
        return ""
    s = int(sec) % 86400
    return f"{s // 3600:02d}{(s % 3600) // 60:02d}:{s % 60:02d}Z"


def render_transcript(records: list[dict]) -> str:
    """Turn the JSONL records into a readable transcript for the reviewer."""
    lines = []
    for r in records:
        k = r["kind"]
        when = _zulu((r.get("state") or {}).get("zulu_sec")) or time.strftime("%H:%M:%S", time.localtime(r["t"]))
        if k == "session_start":
            lines.append(f"SESSION START {time.strftime('%Y-%m-%d %H:%M', time.localtime(r['t']))}: "
                         f"aircraft {r.get('callsign')} ({r.get('type')}), controller model {r.get('chat_model')}")
        elif k == "pilot":
            st = r.get("state", {})
            where = f"{'ground' if st.get('on_ground') else str(st.get('alt_ft')) + ' ft'}, {st.get('gs_kt')} kt"
            lines.append(f"\n[{when}] COM{st.get('tx_com')} {fmt_freq(r.get('khz', 0))} "
                         f"({r.get('facility') or 'no facility'}) [{where}, xpdr {st.get('xpdr')}]")
            lines.append(f"PILOT: {r['text']}")
        elif k == "atc":
            who = r.get("facility", "?")
            if r.get("event"):
                lines.append(f"\n[{when}] (controller-initiated call; trigger: {r.get('trigger', '')})")
            lines.append(f"ATC {who}: {r['text'] if not r.get('silent') else '(no reply)'}")
            if r.get("situation"):
                lines.append("  FACTS given to the controller:\n    " + r["situation"].replace("\n", "\n    "))
        elif k == "atis":
            lines.append(f"\n[{when}] ATIS {r.get('airport')} information {r.get('letter')}: {r.get('text')}")
        elif k == "chatter":
            lines.append(f"\n[{when}] (background chatter, another aircraft on {r.get('facility')}; not the pilot - "
                         "don't grade it)")
            lines += [f"  {ln}" for ln in r.get("lines", [])]
        elif k == "no_facility":
            lines.append(f"\n[{when}] (pilot transmitted on {fmt_freq(r.get('khz', 0))}; no ATC facility there: "
                         f"{r.get('reason', '')})")
    return "\n".join(lines)


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def run(api, model: str, path: Path) -> tuple[str, Path]:
    records = load(path)
    if not any(r["kind"] == "pilot" for r in records):
        raise SystemExit(f"{path} has no pilot transmissions to review")
    transcript = render_transcript(records)
    report = await api.chat(model, [
        {"role": "system", "content": DEBRIEF_PROMPT},
        {"role": "user", "content": f"SESSION TRANSCRIPT:\n{transcript}"},
    ], max_tokens=40000, reasoning_effort="medium", timeout=600)
    if api.last_finish == "length":
        report += "\n\n**(Report cut off: the reviewing model ran out of output tokens.)**\n"
    out = path.with_name(path.stem + "-debrief.md")
    out.write_text(f"# Flight debrief: {path.name}\n\n{report}\n", encoding="utf-8")
    return report, out
