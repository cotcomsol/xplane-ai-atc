"""Minimal OpenRouter client: transcription, chat and speech."""

from __future__ import annotations

import base64
import logging

import httpx

log = logging.getLogger(__name__)

BASE = "https://openrouter.ai/api/v1"


class OpenRouter:
    def __init__(self, api_key: str):
        if not api_key:
            raise RuntimeError("No OpenRouter API key (set OPENROUTER_API_KEY or atc.toml)")
        self.http = httpx.AsyncClient(
            base_url=BASE, timeout=60,
            headers={"Authorization": f"Bearer {api_key}", "X-Title": "X-Plane AI ATC"},
        )
        self.cost = 0.0
        self.last_finish: str | None = None  # finish_reason of the last chat completion

    async def _post(self, path: str, body: dict, timeout: float = 60.0, retry_timeout: float | None = None
                    ) -> httpx.Response:
        """POST; when retry_timeout is set, a request slower than `timeout` is abandoned and sent again
        (OpenRouter providers occasionally stall for 10-20 s)."""
        try:
            r = await self.http.post(path, json=body, timeout=timeout)
        except httpx.TimeoutException:
            if retry_timeout is None:
                raise
            log.warning("OpenRouter %s slow (> %.0fs), retrying", path, timeout)
            r = await self.http.post(path, json=body, timeout=retry_timeout)
        if r.status_code != 200:
            raise RuntimeError(f"OpenRouter {path} {r.status_code}: {r.text[:300]}")
        return r

    async def transcribe(self, wav: bytes, model: str, prompt: str = "") -> str:
        body = {
            "model": model,
            "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"},
            "language": "en",
        }
        if prompt:
            body["prompt"] = prompt  # vocabulary hint; honored by OpenAI transcribe models
        r = await self._post("/audio/transcriptions", body, timeout=8, retry_timeout=20)
        data = r.json()
        self.cost += float((data.get("usage") or {}).get("cost") or 0)
        return (data.get("text") or "").strip()

    async def chat(self, model: str, messages: list[dict], max_tokens: int = 2000,
                   reasoning_effort: str = "none", timeout: float = 60.0, retry_timeout: float | None = None) -> str:
        body = {"model": model, "messages": messages, "max_tokens": max_tokens, "usage": {"include": True}}
        if reasoning_effort == "none":
            body["reasoning"] = {"enabled": False}
        elif reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}
        r = await self._post("/chat/completions", body, timeout=timeout, retry_timeout=retry_timeout)
        data = r.json()
        self.cost += float((data.get("usage") or {}).get("cost") or 0)
        choice = data["choices"][0]
        self.last_finish = choice.get("finish_reason")
        content = (choice["message"].get("content") or "").strip()
        if not content:
            log.warning("Empty model reply (finish_reason=%s)", choice.get("finish_reason"))
        return content

    async def speech(self, model: str, text: str, voice: str, instructions: str = "",
                     speed: float = 1.0) -> tuple[bytes, int]:
        """Returns (16-bit mono PCM bytes, sample rate)."""
        body = {"model": model, "input": text, "voice": voice, "response_format": "pcm"}
        if speed and speed != 1.0:
            body["speed"] = speed
        if instructions:
            body["instructions"] = instructions
        r = await self._post("/audio/speech", body)
        rate = 24000
        for part in r.headers.get("content-type", "").split(";"):
            if part.strip().startswith("rate="):
                rate = int(part.strip()[5:])
        return r.content, rate

    async def speech_stream(self, model: str, text: str, voice: str, instructions: str = "",
                            speed: float = 1.0):
        """Yield (sample rate, 16-bit mono PCM bytes) chunks as they arrive. Chunks always hold
        whole samples."""
        body = {"model": model, "input": text, "voice": voice, "response_format": "pcm"}
        if speed and speed != 1.0:
            body["speed"] = speed
        if instructions:
            body["instructions"] = instructions
        async with self.http.stream("POST", "/audio/speech", json=body) as r:
            if r.status_code != 200:
                await r.aread()
                raise RuntimeError(f"OpenRouter /audio/speech {r.status_code}: {r.text[:300]}")
            rate = 24000
            for part in r.headers.get("content-type", "").split(";"):
                if part.strip().startswith("rate="):
                    rate = int(part.strip()[5:])
            rest = b""
            async for chunk in r.aiter_bytes():
                data = rest + chunk
                cut = len(data) // 2 * 2
                rest = data[cut:]
                if cut:
                    yield rate, data[:cut]

    async def close(self) -> None:
        await self.http.aclose()
