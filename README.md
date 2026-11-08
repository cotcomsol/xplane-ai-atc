# X-Plane AI ATC

Voice air traffic control for X-Plane 12, driven by AI models you pay for per use instead of by
subscription. Hold the PTT button on your yoke, talk to ground, tower, approach or center the way
you would in the real world, and the controller answers on the radio.

A typical hour of flying costs a few cents in AI usage (OpenRouter). The controller's voice runs
locally on your CPU and costs nothing.

> For flight simulation only. Not for real-world aviation, training or navigation.

## What it does

- **Talks like a US controller** (FAA JO 7110.65 phraseology; ICAO phraseology outside the US),
  using the facility that owns the frequency you're transmitting on: Clearance Delivery, Ground,
  Tower, Approach/Departure, Center.
- **Knows the facts from the sim**, not from the model: your position, altitude, heading, squawk,
  callsign and aircraft type, the runway in use from the wind, altimeter, ATIS letter, airport
  frequencies and taxi routes from X-Plane's airport data.
- **Ground:** taxi routes from the airport's taxi network, with runway crossings; Class B/C
  departure instructions; taxi to parking after landing.
- **Tower:** takeoff and landing clearances, departure turns, pattern work with sequencing
  ("number two, follow the Piper on left base", "extend downwind, I'll call your base"), closed
  traffic, touch and goes, go-arounds, runway exits after landing.
- **Approach / Departure:** radar contact, flight following, traffic advisories from X-Plane's AI
  traffic, handoffs, and **instrument approaches** from X-Plane's CIFP data with radar vectors to
  final ("seven miles from PICTO, turn right heading two five zero, maintain six thousand until
  established on the localizer, cleared ILS runway two eight right approach").
- **IFR:** clearances ("cleared to Monterey airport via radar vectors, then as filed..."), readback
  checks, climbs, handoffs, "expect the ILS" and the approach.
- **SimBrief flight plans:** set your SimBrief username and the clearance comes from the plan you filed
  there: "cleared to San Carlos airport via the LOUPE one departure, HRNER transition, then as filed,
  maintain three thousand, expect niner thousand...", with your filed cruise altitude and destination.
- **Cross-country:** Center frequencies from FAA data, sector handoffs, radar service termination.
- **ATIS** broadcasts built from the sim weather when you tune an ATIS frequency.
- **Controller-initiated calls:** handoffs, vectors, landing clearances you forgot to ask for,
  "how do you hear?" when you don't acknowledge, altitude checks.
- **Background radio chatter:** other aircraft working your controller, matched to X-Plane's AI
  traffic when there is any.
- **Post-flight debrief:** a stronger model reviews the session log against the AIM, the
  Pilot/Controller Glossary and 7110.65, for your radio work and for the sim's own mistakes.

## Requirements

- X-Plane 12.1.4 or newer (uses its built-in web API). The web API only accepts connections from
  the same computer, so run this app on the sim PC.
- Windows or Linux, Python 3.11+ and [uv](https://docs.astral.sh/uv/).
- Linux only: PortAudio (`sudo apt install libportaudio2`).
- An [OpenRouter](https://openrouter.ai) API key with a little credit.
- A microphone, and a joystick/yoke button for push-to-talk.

## Setup

### Windows

1. Install uv once, in PowerShell:
   `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. Download or clone this repository onto the sim PC and double-click `run-atc.bat`. The first time
   it creates `atc.toml` and opens it in Notepad.
3. In `atc.toml` set `[xplane] path` (e.g. `"C:/X-Plane 12"`) and `[openrouter] api_key`, and save.
4. Double-click `run-atc.bat` again.

### Linux

```
git clone <this repo> && cd atc
uv sync
cp atc.example.toml atc.toml     # set [xplane] path and [openrouter] api_key
uv run atc run
```

The API key can also come from the `OPENROUTER_API_KEY` environment variable or a `.env` file
(`./.env` or `~/.env`).

### In X-Plane

- Settings > Joystick: bind your PTT button to `sim/operation/contact_atc` (or the command set in
  `[xplane] ptt_command`).
- Turn off X-Plane's own ATC voice if it gets in the way.
- Tune COM1/COM2 to real frequencies; the app answers on whichever radio you transmit on.

### First run

- X-Plane's airport data (all airports, runways, frequencies) is indexed into `cache/`, which takes
  a little while once.
- The local voice model (~120 MB) is downloaded into `cache/kokoro` once.

## Usage

```
uv run atc run            # fly: PTT -> controller
uv run atc monitor        # live aircraft state, tuned frequency and facility, PTT presses
uv run atc devices        # list audio devices (set input_device / output_device in atc.toml)
uv run atc debrief        # review the latest flight (see below)
uv run atc say "Palo Alto Ground, Skyhawk one two three alpha bravo, request taxi"
                          # typed transmission instead of the microphone
uv run atc say --offline --freq 118.6 --lat 37.46 --lon -122.11 "..."
                          # no X-Plane needed: fake aircraft state
uv run atc datarefs tcas  # search X-Plane's dataref names
```

On Windows, `run-atc.bat` runs `atc run`; `run-atc.bat monitor` etc. run the other commands.

Tips:

- Say your callsign the way it's set in X-Plane (or set `[pilot] callsign` in `atc.toml`).
- With `[simbrief] username` set, just ask clearance delivery (or ground) for your clearance ("ready to
  copy"); the latest plan you generated on SimBrief is fetched at startup and again when you call, so
  re-filing there before you call is enough.
- Without a SimBrief plan, say "VFR" or "IFR" and your destination on your first call ("VFR to Monterey", "IFR to Monterey,
  ready to copy") so the controllers know your plan.
- Ask approach for an approach by name: "request the ILS runway two eight right".

## Configuration

Everything is in `atc.toml` (see `atc.example.toml` for all options and comments). The defaults:

| Step | Default | Typical time |
|---|---|---|
| Speech-to-text | `openai/gpt-4o-mini-transcribe` via OpenRouter | ~0.6-0.9 s |
| Controller | `openai/gpt-5.6-luna`, reasoning off, via OpenRouter | ~1.1 s |
| Voice | Kokoro-82M on your CPU (`voice_engine = "local"`) | ~0.3 s to first audio |
| Debrief | `anthropic/claude-opus-5.5` via OpenRouter | ~2 min, ~$0.15-0.45 |

- `[audio] voice_engine = "openrouter"` uses OpenRouter's voice models instead of the local one (the
  local voice also falls back to it if it can't load). The voice has to read spelled-out numbers
  literally ("wind three zero zero", not "wind three hundred"): Kokoro, Gemini TTS and Grok TTS do;
  Microsoft MAI and ElevenLabs voices don't.
- `[audio] voice_threads`: CPU threads for the local voice (lower it if X-Plane stutters while ATC talks).
- `[simbrief] username`: your SimBrief username or numeric Pilot ID (simbrief.com Account Settings; not
  your email). The plan is used when it departs from the airport you're at. Its departure procedure is
  checked against X-Plane's CIFP data: one that doesn't serve the runway in use is amended to radar
  vectors to the first fix on your route.
- `[chatter] mode`: `always` (AI traffic, else made-up aircraft), `traffic` (AI traffic only) or `off`.
- `[xplane] cifp_dirs`: extra folders with CIFP approach data (X-Plane's own `Custom Data/CIFP`, then
  `Resources/default data/CIFP`, are searched as well).

## Post-flight debrief

`atc run` logs every exchange to `logs/flight-<date>-<time>.jsonl`: your transcribed calls, the
controller's replies and silences, controller-initiated calls, ATIS, and the facts the controller was
given each time. `uv run atc debrief` (latest log) or `uv run atc debrief logs/<file>.jsonl` sends it to
the debrief model and saves a Markdown report next to the log:

- **Pilot radio work:** transmissions that could be better, with corrected phrasing, references
  (AIM 4-2, P/CG) and the top things to practice.
- **Controller and simulator review:** replies a real controller wouldn't make, each tagged
  PHRASING / FACTS / CAPABILITY / TRANSCRIPTION with a suggested fix.
- **Transcription problems.**

## How it works

```
yoke PTT -> X-Plane command -> web API (websocket) -> record mic
  -> speech-to-text -> controller (facts computed in code + language model) -> voice
  -> VHF radio effect -> speakers
```

The language model only phrases the reply. Everything a controller must get right is computed in
code and handed to it as a SITUATION block: which facility owns the frequency, the runway in use,
magnetic wind, altimeter, taxi route, squawk and radar identification, pattern position and landing
sequence, vectors and approach clearances, handoff points, whether the pilot's readback matches what
was assigned. Small, fast models phrase well but get facts wrong; this split keeps replies around a
second and correct.

Data sources:

- Airports, runways, frequencies and taxi networks: X-Plane's `apt.dat` (enabled custom scenery packs
  first, in `scenery_packs.ini` order, then the global airports), cached in `cache/`.
- Instrument approaches: X-Plane's CIFP files.
- Class B/C airports: FAA airspace data (`src/atc/airspace.py`).
- Center radio sites and frequencies: FAA NASR (`src/atc/data/centers.json`).
- Traffic: X-Plane's TCAS targets (AI and multiplayer aircraft).
- Filed flight plans: SimBrief's public flight plan fetcher (`src/atc/flightplan.py`); departure
  procedures from X-Plane's CIFP files.

Source layout: `xplane.py` (web API client), `state.py` (aircraft state), `airports.py`,
`taxi.py`, `procedures.py` (CIFP approaches and vectoring), `pattern.py`, `traffic.py`,
`flightplan.py` (SimBrief), `controller.py` (facts, events, prompt), `atis.py`, `chatter.py`, `local_tts.py`, `audio.py`
(recording, playback, radio effect), `openrouter.py`, `debrief.py`, `main.py` (commands).

## Limitations

- Without a SimBrief plan, IFR routes are always "radar vectors, then as filed".
- Filed arrival procedures (STARs) aren't used yet; arrivals get vectors to the approach.
- No terrain or minimum vectoring altitudes: vectoring altitudes come from the approach procedure.
- Center handoffs need to know your destination (say it on your first call).
- Speech recognition mishears sometimes, especially callsigns; the controller will ask you to say
  again, and the debrief lists what it misheard.
- The controller is a small language model and occasionally phrases something oddly. The debrief is
  the way to find these; issues with a debrief excerpt are welcome.

## Development

```
uv run python -m unittest discover -s tests
```

`tests/fixtures/xp` is a tiny fake X-Plane folder with approximate data for a few airports.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).

---

Planning a real flight? Check out [CAVU Route](https://cavuroute.app/), our weather planning app.
