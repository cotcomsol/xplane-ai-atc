"""Command line entry point.

  atc monitor        show live aircraft state and the facility on the tuned frequency
  atc run            full loop: PTT -> speech-to-text -> controller -> voice
  atc say "text"     send a typed transmission (no microphone) and hear the reply
  atc devices        list audio devices
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
import zlib

from . import config as config_mod
from .airports import AirportDB, Facility
from .controller import SILENCE, Controller, clean_reply, spoken_facility
from .phraseology import spell
from .pronounce import for_tts
from .flightlog import FlightLog, snapshot
from .state import AircraftState, fmt_freq
from .xplane import XPlane

log = logging.getLogger("atc")

NO_ATC_KINDS = ("CTAF", "ATIS")
EVENT_MAX_DEFER_S = 45.0  # a controller-initiated call held back while the pilot talked; dropped when older
REMOTE_VOICE_TIMEOUT_S = 6.0  # OpenRouter TTS: retry when the first audio takes longer


def _facility(db: AirportDB | None, s: AircraftState) -> Facility | None:
    return db.facility_for(s.tx_khz, s.lat, s.lon) if db else None


def _status(s: AircraftState, fac: Facility | None, callsign: str) -> str:
    where = f"{fac.airport['id']} {fac.kind} ({fac.name})" if fac else "no facility"
    return (f"{callsign or '?'} {s.icao_type} | {s.lat:.4f},{s.lon:.4f} {s.alt_ft:.0f}ft {s.gs_kt:.0f}kt "
            f"hdg {s.hdg_mag:03.0f} {'GND' if s.on_ground else 'AIR'} | COM1 {fmt_freq(s.com1_khz)} "
            f"COM2 {fmt_freq(s.com2_khz)} TX COM{s.tx_com} -> {where} | xpdr {s.xpdr_code:04d}")


async def cmd_monitor(cfg) -> None:
    xp = XPlane(cfg.xplane_host, cfg.xplane_port)
    xp.on_command(cfg.ptt_command, lambda on: print(f"  PTT {'DOWN' if on else 'UP'}"))
    db = AirportDB.load(cfg.xplane_path, cfg.cache_dir)
    task = asyncio.create_task(xp.run())
    try:
        while True:
            await asyncio.sleep(1)
            if xp.connected and xp.values:
                s = AircraftState.from_values(xp.values)
                print(_status(s, _facility(db, s), cfg.callsign or s.tailnum))
    finally:
        task.cancel()


def readback_window(reply: str) -> float:
    """Seconds to leave the pilot after a transmission before the controller starts another call:
    reaction time plus roughly the time to read it back (an approach clearance gets ~13 s)."""
    return min(14.0, 3.0 + 0.3 * len(reply.split()))


class Session:
    """Everything needed to turn a pilot transmission into a controller reply."""

    def __init__(self, cfg):
        from .openrouter import OpenRouter

        self.cfg = cfg
        self.db = AirportDB.load(cfg.xplane_path, cfg.cache_dir)
        self.ctl = Controller(self.db, xplane_path=cfg.xplane_path, cifp_dirs=cfg.cifp_dirs)
        self.api = OpenRouter(cfg.api_key)
        self._player = None
        self._lock = asyncio.Lock()
        self.log = None  # FlightLog when recording a session
        self.local = None  # LocalVoice when the voice is generated on this computer
        self._plan_checked = 0.0  # monotonic time of the last SimBrief fetch
        self.ptt_presses = 0  # counts PTT key-ups, so a controller call started before one isn't played over it
        self.quiet_until = 0.0  # no controller-initiated calls or chatter before this (monotonic): readback time
        self.deferred_event: tuple[str, float] | None = None  # (event, monotonic time) to retry when quiet
        self.transmitting = lambda: False  # is the pilot's PTT down right now? (set by `atc run`)
        if cfg.voice_engine == "local":
            from .local_tts import LocalVoice

            if LocalVoice.available():
                self.local = LocalVoice(cfg.cache_dir, cfg.voice_threads)
            else:
                log.warning("Local voice unavailable (kokoro-onnx not installed); using OpenRouter %s", cfg.tts_model)
        self.stt_local = None  # LocalSTT when speech-to-text runs on this computer
        if cfg.stt_engine == "local":
            from .local_stt import LocalSTT

            if LocalSTT.available():
                self.stt_local = LocalSTT(cfg.cache_dir, cfg.stt_threads)
            else:
                log.warning("Local speech-to-text unavailable (onnx-asr not installed); using OpenRouter %s",
                            cfg.stt_model)

    @property
    def stt_name(self) -> str:
        return self.stt_local.name if self.stt_local else self.cfg.stt_model

    async def prepare_stt(self) -> None:
        """Download/load the local speech-to-text model up front."""
        if self.stt_local:
            log.info("Loading local speech-to-text...")
            try:
                await asyncio.to_thread(self.stt_local.load)
            except Exception as e:  # noqa: BLE001
                log.warning("Local speech-to-text failed to load (%s); using OpenRouter %s", e, self.cfg.stt_model)
                self.stt_local = None

    async def transcribe(self, samples, wav: bytes, hint: str, icao_type: str = "") -> str:
        """Pilot audio -> text, locally when configured (OpenRouter if that fails)."""
        if self.stt_local:
            try:
                from .local_stt import fix_make
                from .phraseology import type_name

                text = await asyncio.to_thread(self.stt_local.transcribe, samples)
                return fix_make(text, type_name(icao_type)) if icao_type else text
            except Exception as e:  # noqa: BLE001
                log.warning("Local speech-to-text failed (%s); using OpenRouter from now on", e)
                self.stt_local = None
        return await self.api.transcribe(wav, self.cfg.stt_model, hint)

    async def prepare_voice(self) -> None:
        """Download/load the local voice up front so the first transmission doesn't wait for it."""
        if self.local:
            log.info("Loading the local voice...")
            try:
                await asyncio.to_thread(self.local.load)
            except Exception as e:  # noqa: BLE001
                log.warning("Local voice failed to load (%s); using OpenRouter %s", e, self.cfg.tts_model)
                self.local = None

    async def refresh_flight_plan(self) -> None:
        """Fetch the latest SimBrief plan (keeps the previous one if the fetch fails)."""
        if not self.cfg.simbrief_user:
            return
        from .flightplan import fetch_simbrief

        self._plan_checked = time.monotonic()
        try:
            plan = await fetch_simbrief(self.cfg.simbrief_user, timeout=4)
        except Exception as e:  # noqa: BLE001 - a missing plan must never stop the radio
            log.warning("SimBrief flight plan not loaded: %s", e)
            return
        old = self.ctl.flight_plan
        if old and old.as_dict() == plan.as_dict():
            return
        self.ctl.set_flight_plan(plan)
        log.info("Filed flight plan (SimBrief): %s", plan.summary)
        if self.log:
            self.log.write("flight_plan", source=plan.source, summary=plan.summary, plan=plan.as_dict())

    @property
    def player(self):
        if self._player is None:
            from .audio import Player
            self._player = Player(self.cfg.output_device)
        return self._player

    @property
    def busy(self) -> bool:
        """True while a pilot transmission is being answered."""
        return self._lock.locked()

    def voice_for(self, fac: Facility | None) -> str:
        voices = self.cfg.tts_voices or ["alloy"]
        return voices[zlib.crc32((fac.key if fac else "").encode()) % len(voices)]

    def stt_hint(self, s: AircraftState) -> str:
        """Context for speech-to-text so names and callsigns come out right."""
        parts = ["Pilot radio call to air traffic control."]
        fac = _facility(self.db, s)
        if fac:
            parts.append(f"Facility: {spoken_facility(fac)}, {fac.airport['name']} ({fac.airport['id']}).")
        reg = self.cfg.callsign or s.tailnum
        if reg:
            parts.append(f"Callsign {reg} ({spell(reg)}).")
        if s.icao_type:
            parts.append(f"Aircraft type {s.icao_type}.")
        parts.append("Words: runway, taxi, holding point, hold short, QNH, altimeter, squawk, niner, "
                     "information, ready for departure, request, VFR, flight following.")
        return " ".join(parts)

    def quiet(self) -> bool:
        return time.monotonic() >= self.quiet_until

    async def respond(self, transcript: str, s: AircraftState, play: bool = True,
                      out_wav: str | None = None, event: bool = False) -> str | None:
        async with self._lock:
            t0 = time.monotonic()
            presses = self.ptt_presses
            fac = _facility(self.db, s)
            if fac is None or fac.kind in NO_ATC_KINDS:
                what = f"{fac.airport['id']} {fac.kind}" if fac else "no ATC facility found"
                log.info("Nobody answers on %s (%s)", fmt_freq(s.tx_khz), what)
                if self.log and not event:
                    self.log.write("no_facility", khz=s.tx_khz, reason=what, state=snapshot(s))
                return None
            if (self.cfg.simbrief_user and not event and s.on_ground and fac.kind in ("DELIVERY", "GROUND")
                    and not self.ctl.ifr_clearance and time.monotonic() - self._plan_checked > 60):
                await self.refresh_flight_plan()  # pick up a plan re-filed since startup
            callsign = self.cfg.callsign or s.tailnum
            situation = self.ctl.situation(s, fac, callsign, "" if event else transcript)
            log.debug("SITUATION:\n%s", situation)
            reply = await self.api.chat(self.cfg.chat_model,
                                        self.ctl.messages(fac, situation, transcript, event=event),
                                        reasoning_effort=self.cfg.reasoning_effort, timeout=8, retry_timeout=20)
            reply = clean_reply(reply)
            t1 = time.monotonic()
            if event and self.ptt_presses != presses:
                # the pilot keyed up while this call was being prepared: don't talk over them; retry once quiet
                log.info("(ATC-initiated call held: the pilot is transmitting)")
                self.deferred_event = (transcript, t0)
                return None
            self.ctl.remember(fac, transcript, reply, event=event)
            if self.log:
                self.log.write("atc", facility=f"{fac.airport['id']} {fac.kind}", text=reply,
                               silent=(not reply or SILENCE in reply), event=event,
                               trigger=transcript if event else None, situation=situation, state=snapshot(s))
            if not reply or SILENCE in reply:
                log.info("ATC (%s): <no reply>  [%.1fs]", fac.name, t1 - t0)
                if not event:
                    self.quiet_until = time.monotonic() + 2.0  # readback taken: a short gap, then free to talk
                return None
            log.info("ATC (%s): %s  [model %.1fs]", fac.name, reply, t1 - t0)
            if play:  # the pilot keyed up again while this was prepared: answer once they let go
                waited = time.monotonic()
                while self.transmitting() and time.monotonic() - waited < 60:
                    await asyncio.sleep(0.1)
            if play or out_wav:
                await self.speak(reply, fac, play=play, out_wav=out_wav)
            self.quiet_until = time.monotonic() + readback_window(reply)
            return reply

    async def _remote_chunks(self, text: str, voice: str):
        """OpenRouter TTS as (rate, float32) chunks; retried once if the first audio is slow."""
        from .audio import pcm16_to_float

        for attempt in (1, 2):
            gen = self.api.speech_stream(self.cfg.tts_model, for_tts(text, self.cfg.tts_model), voice,
                                         self.cfg.tts_instructions, self.cfg.tts_speed)
            try:
                first = await asyncio.wait_for(anext(gen), timeout=REMOTE_VOICE_TIMEOUT_S * attempt)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError:
                await gen.aclose()
                if attempt == 2:
                    raise RuntimeError(f"voice took over {REMOTE_VOICE_TIMEOUT_S * 2:.0f}s") from None
                log.warning("  [voice slow (> %.0fs), retrying]", REMOTE_VOICE_TIMEOUT_S)
                continue
            yield first[0], pcm16_to_float(first[1])
            async for rate, pcm in gen:
                yield rate, pcm16_to_float(pcm)
            return

    async def _voice_chunks(self, text: str, voice: str):
        if self.local:
            try:
                async for chunk in self.local.stream(text, voice, self.cfg.tts_speed):
                    yield chunk
                return
            except Exception as e:  # noqa: BLE001 - fall back to the network voice
                log.warning("Local voice failed (%s); using OpenRouter from now on", e)
                self.local = None
        async for chunk in self._remote_chunks(text, voice):
            yield chunk

    async def speak_streaming(self, text: str, fac: Facility | None, voice: str | None = None) -> None:
        """Play the voice while the rest of it is still being generated."""
        from .audio import RadioStream

        t0 = time.monotonic()
        out = fx = None
        seconds = 0.0
        try:
            async for rate, x in self._voice_chunks(text, voice or self.voice_for(fac)):
                if out is None:
                    out = self.player.stream(rate)
                    fx = RadioStream(rate) if self.cfg.radio_effect else None
                    log.info("  [voice started after %.1fs%s]", time.monotonic() - t0,
                             " (local)" if self.local else "")
                seconds += x.size / rate
                out.feed(fx.process(x) if fx else x)
                if not out.active:  # cut off (pilot keyed the mic)
                    return
            if out is None:
                return
            if fx:
                out.feed(fx.tail())
            out.finish()
            log.info("  [%.1fs of audio; session cost $%.4f]", seconds, self.api.cost)
            await asyncio.to_thread(out.wait)
        except Exception:
            if out:
                out.abort()
            raise

    async def render(self, text: str, fac: Facility | None, voice: str | None = None):
        """Text -> (samples, rate) in the facility's voice, with the radio effect."""
        from .audio import pcm16_to_float, radio_effect

        import re

        import numpy as np

        t0 = time.monotonic()
        voice = voice or self.voice_for(fac)
        samples = rate = None
        if self.local:
            try:
                from .local_tts import RATE

                samples, rate = await asyncio.to_thread(self.local.render, text, voice, self.cfg.tts_speed), RATE
            except Exception as e:  # noqa: BLE001
                log.warning("Local voice failed (%s); using OpenRouter from now on", e)
                self.local = None
        if samples is None:
            # Long texts (ATIS) are voiced sentence by sentence in parallel: much faster
            chunks = [c for c in re.split(r"(?<=\.)\s+", text) if c.strip()] if len(text) > 160 else [text]
            results = await asyncio.gather(*(
                self.api.speech(self.cfg.tts_model, for_tts(c, self.cfg.tts_model), voice,
                                self.cfg.tts_instructions, self.cfg.tts_speed) for c in chunks))
            rate = results[0][1]
            gap = np.zeros(int(0.25 * rate), dtype=np.float32)
            parts = []
            for pcm, _ in results:
                parts += [pcm16_to_float(pcm), gap]
            samples = np.concatenate(parts[:-1])
        log.info("  [voice generated in %.1fs, %.1fs of audio; session cost $%.4f]",
                 time.monotonic() - t0, samples.size / rate, self.api.cost)
        if self.cfg.radio_effect:
            samples = radio_effect(samples, rate)
        return samples, rate

    async def speak(self, text: str, fac: Facility | None, play: bool = True, out_wav: str | None = None,
                    voice: str | None = None):
        from .audio import to_wav

        if play and not out_wav and self.cfg.stream_voice:
            await self.speak_streaming(text, fac, voice)
            return
        samples, rate = await self.render(text, fac, voice)
        if out_wav:
            with open(out_wav, "wb") as fh:
                fh.write(to_wav(samples, rate))
            log.info("Wrote %s", out_wav)
        if play:
            await asyncio.to_thread(self.player.play, samples, rate)


def _atis_facility(db: AirportDB | None, s: AircraftState) -> Facility | None:
    """ATIS on the transmitting radio, or on the other radio if its receive switch is on."""
    if not db:
        return None
    other = 2 if s.tx_com == 1 else 1
    radios = [s.tx_com] + ([other] if (s.rx_com2 if other == 2 else s.rx_com1) else [])
    for com in radios:
        khz = s.com1_khz if com == 1 else s.com2_khz
        fac = db.facility_for(khz, s.lat, s.lon)
        if fac and fac.kind == "ATIS":
            return fac
    return None


async def cmd_run(cfg) -> None:
    from .audio import MIC_RATE, Recorder, to_wav

    xp = XPlane(cfg.xplane_host, cfg.xplane_port)
    sess = Session(cfg)
    await sess.prepare_voice()
    await sess.prepare_stt()
    sess.log = FlightLog(cfg.log_dir)
    log.info("Logging this session to %s", sess.log.path)
    await sess.refresh_flight_plan()
    session_started = False
    rec = Recorder(cfg.input_device)
    sess.transmitting = lambda: rec.recording
    pending: set[asyncio.Task] = set()
    last_status = [""]
    audio_dir = sess.log.path.with_name(sess.log.path.stem + "-audio") if cfg.save_recordings else None
    n_rec = [0]

    async def handle(samples, s: AircraftState):
        secs = samples.size / MIC_RATE
        if secs < 0.4:
            log.info("(PTT too short, ignored)")
            return
        try:
            wav, hint = to_wav(samples), sess.stt_hint(s)
            audio = None
            if audio_dir:  # keep the recording for speech-to-text benchmarks (tools/stt_bench.py)
                n_rec[0] += 1
                audio_dir.mkdir(parents=True, exist_ok=True)
                (audio_dir / f"{n_rec[0]:03d}.wav").write_bytes(wav)
                audio = f"{audio_dir.name}/{n_rec[0]:03d}.wav"
            t0 = time.monotonic()
            text = await sess.transcribe(samples, wav, hint, s.icao_type)
            stt_s = round(time.monotonic() - t0, 2)
            log.info("PILOT (COM%d %s): %s  [%.1fs]", s.tx_com, fmt_freq(s.tx_khz),
                     text or "<nothing heard>", stt_s)
            if audio:
                sess.log.write("recording", audio=audio, secs=round(secs, 2), stt_model=sess.stt_name,
                               hint=hint, text=text, stt_s=stt_s)
            if text:
                nonlocal session_started
                if not session_started:
                    session_started = True
                    sess.log.write("session_start", callsign=cfg.callsign or s.tailnum, type=s.icao_type,
                                   chat_model=cfg.chat_model)
                fac = _facility(sess.db, s)
                sess.log.write("pilot", text=text, khz=s.tx_khz, state=snapshot(s),
                               facility=f"{fac.airport['id']} {fac.kind}" if fac else None)
                await sess.respond(text, s)
        except Exception as e:  # noqa: BLE001 - keep the loop alive
            log.error("Transmission failed: %s", e)

    def on_ptt(active: bool):
        if active:
            sess.ptt_presses += 1
            sess.player.stop()  # pilot keys up over ATC: cut ATC off
            try:
                rec.start()
                log.info("PTT down - recording")
            except Exception as e:  # noqa: BLE001
                log.error("Microphone error: %s", e)
        else:
            samples = rec.stop()
            if not xp.values:
                return
            t = asyncio.get_running_loop().create_task(handle(samples, AircraftState.from_values(xp.values)))
            pending.add(t)
            t.add_done_callback(pending.discard)

    atis_playing = [None, None]  # (facility key, letter) being broadcast, and its audio

    async def atis_tick():
        from . import atis

        if not (xp.connected and xp.values):
            return
        s = AircraftState.from_values(xp.values)
        fac = _atis_facility(sess.db, s)
        if fac is None:
            if atis_playing[0]:
                atis_playing[0] = None
                if not sess.busy:  # (a controller's reply may be playing by now: leave that alone)
                    sess.player.stop()
            return
        if rec.recording or sess.busy:
            return
        text, letter = atis.build(fac, s)
        key = (fac.key, letter)
        if key != atis_playing[0]:
            sess.player.stop()
            log.info("ATIS %s: %s", fac.airport["id"], text)
            sess.log.write("atis", airport=fac.airport["id"], letter=letter, text=text, state=snapshot(s))
            atis_playing[1] = await sess.render(text, fac)
            atis_playing[0] = key
        if not sess.player.busy():
            await asyncio.sleep(1.5)  # gap between repeats
            if atis_playing[0] == key and not rec.recording and not sess.busy:
                sess.player.start(*atis_playing[1])

    async def atis_loop():
        while True:
            await asyncio.sleep(0.5)
            try:
                await atis_tick()
            except Exception as e:  # noqa: BLE001 - one failed broadcast must not end ATIS for the session
                log.warning("ATIS failed (%s); trying again shortly", e)
                atis_playing[0] = None
                await asyncio.sleep(5)

    async def initiate(ev: str, s: AircraftState, since: float):
        """A controller-initiated call; if it fails (network, model) it is tried again when quiet."""
        try:
            await sess.respond(ev, s, event=True)
        except Exception as e:  # noqa: BLE001
            log.error("ATC-initiated call failed (%s); will retry", e)
            sess.deferred_event = (ev, since)
            sess.quiet_until = time.monotonic() + 5.0

    from .chatter import Chatter

    chatter = Chatter(sess, cfg.chatter_mode, cfg.chatter_min_gap, cfg.chatter_max_gap)

    def is_idle(ignore_lock: bool = False) -> bool:
        return not rec.recording and (ignore_lock or not sess.busy) and not pending and sess.quiet()

    xp.on_command(cfg.ptt_command, on_ptt)
    task = asyncio.create_task(xp.run())
    atis_task = asyncio.create_task(atis_loop())
    log.info("Bind your yoke PTT button to the X-Plane command '%s'. Ctrl+C to quit.", cfg.ptt_command)
    try:
        while True:
            await asyncio.sleep(1)
            if xp.connected and xp.values:
                s = AircraftState.from_values(xp.values)
                fac = _facility(sess.db, s)
                st = f"COM{s.tx_com} {fmt_freq(s.tx_khz)} -> " + (
                    f"{fac.airport['id']} {fac.kind} \"{fac.name}\"" if fac else "no facility")
                if st != last_status[0]:
                    last_status[0] = st
                    log.info("Tuned: %s", st)
                if s.paused:  # sim paused: nobody calls, and the timers don't run
                    sess.ctl.shift_timers(1.0)
                    sess.quiet_until += 1.0
                    chatter.next_t += 1.0
                    continue
                if chatter.due(fac, busy=not is_idle()):
                    chatter.start(s, fac, is_idle)
                if not rec.recording and not sess.busy and not pending and sess.quiet():
                    ev = None
                    since = time.monotonic()
                    if sess.deferred_event:  # held back while the pilot talked, or it failed
                        ev, since = sess.deferred_event
                        sess.deferred_event = None
                        if time.monotonic() - since > EVENT_MAX_DEFER_S:
                            log.warning("(ATC-initiated call dropped, too old: %s)", ev.split(".")[0])
                            ev, since = None, time.monotonic()
                    ev = ev or sess.ctl.pending_event(s, fac)
                    if ev:
                        log.info("(ATC-initiated call: %s)", ev.split(".")[0])
                        t = asyncio.create_task(initiate(ev, s, since))
                        pending.add(t)
                        t.add_done_callback(pending.discard)
    finally:
        task.cancel()
        atis_task.cancel()
        sess.log.close()
        log.info("Session AI cost: $%.4f", sess.api.cost)
        if session_started:
            log.info("Post-flight review: uv run atc debrief   (log: %s)", sess.log.path)


async def cmd_transcribe(cfg, args) -> None:
    """Transcribe WAV files with the configured engine (or --engine), with the time each took."""
    import wave

    import numpy as np
    from scipy.signal import resample_poly

    from .audio import MIC_RATE, to_wav

    if args.engine:
        cfg.stt_engine = args.engine
    sess = Session(cfg)
    await sess.prepare_stt()
    for path in args.wavs:
        with wave.open(path) as w:
            rate, ch = w.getframerate(), w.getnchannels()
            x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
        if ch > 1:
            x = x.reshape(-1, ch).mean(axis=1)
        if rate != MIC_RATE:
            x = resample_poly(x, MIC_RATE, rate).astype(np.float32)
        t0 = time.monotonic()
        text = await sess.transcribe(x, to_wav(x), "Pilot radio call to air traffic control.")
        print(f"{path}: [{time.monotonic() - t0:.2f}s {sess.stt_name}] {text}")
    await sess.api.close()


async def cmd_say(cfg, args) -> None:
    sess = Session(cfg)
    await sess.refresh_flight_plan()
    if not args.no_play or args.out:
        await sess.prepare_voice()
    s = None
    if not args.offline:
        xp = XPlane(cfg.xplane_host, cfg.xplane_port)
        task = asyncio.create_task(xp.run())
        for _ in range(30):
            await asyncio.sleep(0.2)
            if xp.connected and xp.values:
                await asyncio.sleep(0.3)
                s = AircraftState.from_values(xp.values)
                break
        task.cancel()
    if s is None:
        log.info("Using a fake aircraft state (X-Plane not connected or --offline)")
        s = AircraftState(
            lat=args.lat, lon=args.lon, alt_ft=args.alt, agl_ft=0 if args.alt < 100 else args.alt,
            gs_kt=0 if args.alt < 100 else 100, ias_kt=0, hdg_mag=310, vs_fpm=0, on_ground=args.alt < 100,
            com1_khz=int(round(args.freq * 1000)), com2_khz=0, tx_com=1, xpdr_code=1200,
            xpdr_mode="alt (mode C)", tailnum=args.callsign, icao_type="C172",
            altimeter_inhg=29.92, wind_dir=args.wind_dir, wind_kt=args.wind_kt,
        )
    fac = _facility(sess.db, s)
    print(_status(s, fac, cfg.callsign or s.tailnum))
    if args.show_situation:
        print(sess.ctl.situation(s, fac, cfg.callsign or s.tailnum))
    if fac and fac.kind == "ATIS":
        from . import atis

        text, _ = atis.build(fac, s)
        print(text)
        await sess.speak(text, fac, play=not args.no_play, out_wav=args.out)
    else:
        await sess.respond(args.text, s, play=not args.no_play, out_wav=args.out)
    await sess.api.close()


async def cmd_debrief(cfg, args) -> None:
    from pathlib import Path

    from . import debrief
    from .flightlog import latest
    from .openrouter import OpenRouter

    path = Path(args.log) if args.log else latest(cfg.log_dir)
    if not path or not path.exists():
        raise SystemExit(f"No flight logs in {cfg.log_dir}/ (they are written by `atc run`)")
    model = args.model or cfg.debrief_model
    log.info("Reviewing %s with %s ...", path, model)
    api = OpenRouter(cfg.api_key)
    t0 = time.monotonic()
    report, out = await debrief.run(api, model, path)
    await api.close()
    print(report)
    log.info("Saved %s  [%.0fs, cost $%.3f]", out, time.monotonic() - t0, api.cost)


async def cmd_datarefs(cfg, text: str) -> None:
    xp = XPlane(cfg.xplane_host, cfg.xplane_port)
    for d in await xp.search_datarefs(text):
        print(f"{d['name']}  ({d.get('value_type')})")


def cmd_devices() -> None:
    import sounddevice as sd

    print(sd.query_devices())


def main() -> None:
    p = argparse.ArgumentParser(prog="atc", description="AI ATC for X-Plane 12")
    p.add_argument("-c", "--config", help="configuration file (default: atc.toml)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("monitor")
    sub.add_parser("run")
    sub.add_parser("devices")
    db_ = sub.add_parser("debrief", help="post-flight review of a session log by a stronger model")
    db_.add_argument("log", nargs="?", help="flight log (default: the latest in logs/)")
    db_.add_argument("--model", help="override debrief_model")
    dp = sub.add_parser("datarefs", help="search X-Plane dataref names")
    dp.add_argument("text")
    tp = sub.add_parser("transcribe", help="speech-to-text on WAV files (e.g. saved recordings), with timing")
    tp.add_argument("wavs", nargs="+")
    tp.add_argument("--engine", choices=["openrouter", "local"], help="override [audio] stt_engine")
    sp = sub.add_parser("say")
    sp.add_argument("text", nargs="?", default="", help="what the pilot says (not needed for ATIS)")
    sp.add_argument("--offline", action="store_true", help="don't try to connect to X-Plane")
    sp.add_argument("--lat", type=float, default=37.4611)
    sp.add_argument("--lon", type=float, default=-122.1150)
    sp.add_argument("--alt", type=float, default=7)
    sp.add_argument("--freq", type=float, default=125.0, help="COM1 MHz for the fake state")
    sp.add_argument("--callsign", default="N123AB")
    sp.add_argument("--wind-dir", type=int, default=300)
    sp.add_argument("--wind-kt", type=int, default=8)
    sp.add_argument("--no-play", action="store_true")
    sp.add_argument("--out", help="also write the reply audio to this WAV file")
    sp.add_argument("--show-situation", action="store_true")
    args = p.parse_args()
    import faulthandler

    faulthandler.enable()  # a crash inside native code (onnxruntime, PortAudio) prints a traceback instead of vanishing

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if args.config:
        from pathlib import Path

        if not Path(args.config).is_file():
            raise SystemExit(f"Config file not found: {args.config}")
    cfg = config_mod.load(args.config or "atc.toml")
    try:
        if args.cmd == "monitor":
            asyncio.run(cmd_monitor(cfg))
        elif args.cmd == "run":
            asyncio.run(cmd_run(cfg))
        elif args.cmd == "transcribe":
            asyncio.run(cmd_transcribe(cfg, args))
        elif args.cmd == "say":
            asyncio.run(cmd_say(cfg, args))
        elif args.cmd == "debrief":
            asyncio.run(cmd_debrief(cfg, args))
        elif args.cmd == "datarefs":
            asyncio.run(cmd_datarefs(cfg, args.text))
        elif args.cmd == "devices":
            cmd_devices()
    except KeyboardInterrupt:
        pass
    except Exception as e:  # noqa: BLE001
        import httpx

        if isinstance(e, httpx.ConnectError):
            try:
                host, port = e.request.url.host, e.request.url.port
            except RuntimeError:  # no request attached to the error
                host = port = None
            if host is None or (host == cfg.xplane_host and port == cfg.xplane_port):
                raise SystemExit(f"Can't reach X-Plane at {cfg.xplane_host}:{cfg.xplane_port} "
                                 "(is it running with a flight loaded?)") from None
            raise SystemExit(f"Can't reach {host} ({e}); check the internet connection") from None
        raise


if __name__ == "__main__":
    main()
