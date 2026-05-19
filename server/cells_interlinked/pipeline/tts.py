"""OpenAI gpt-4o-mini-tts proxy for /chat voice mode.

Gemma emits a `<speech>...</speech><voice>...</voice>` envelope each
turn (see chat_loop.py); this module ships the parsed pieces to
OpenAI's standalone TTS endpoint and returns the rendered audio bytes
to the browser. The `instructions` field is the prompt-steering
surface — free-text direction passes through verbatim, which is the
whole reason this provider is the right fit (vs. providers that
expose voice-preset enums only).

The browser never sees the API key; the server holds it. Audio comes
back as MP3 for trivial `<audio>` playback on the laptop side.
"""

from __future__ import annotations

import logging
import re

import aiohttp

from ..config import settings

logger = logging.getLogger(__name__)

OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech"

_SPEECH_RE = re.compile(r"<speech>\s*(.*?)\s*</speech>", re.DOTALL | re.IGNORECASE)
_VOICE_RE = re.compile(r"<voice>\s*(.*?)\s*</voice>", re.DOTALL | re.IGNORECASE)

# Cap what we send to OpenAI per call. gpt-4o-mini-tts will happily
# generate minutes of audio if you let it; chat turns are short by
# design, so anything past 2000 chars is almost certainly a runaway.
_TEXT_CAP = 2000
_STYLE_CAP = 800

# Default style when Gemma forgot or the ablated pass corrupted the
# tags. Neutral so the audio plays without bias either direction.
_DEFAULT_STYLE = "Speak in a calm, neutral, clearly enunciated tone."


def parse_voice_envelope(text: str) -> tuple[str, str]:
    """Pull `<speech>` and `<voice>` blocks out of a model output.

    Designed to be tolerant: missing tags, mixed case, surrounding
    whitespace, partial envelopes. The ablated pass can drift far off
    the formatting nudge — we still want to render *something* rather
    than swallow the turn.

    Returns (speech_text, voice_style). Falls back to:
    - speech = everything outside the tags if `<speech>` is absent
    - style = a neutral default if `<voice>` is absent
    """
    if not text:
        return "", _DEFAULT_STYLE

    speech_m = _SPEECH_RE.search(text)
    voice_m = _VOICE_RE.search(text)

    if speech_m:
        speech = speech_m.group(1).strip()
    else:
        # No <speech> tag — strip any <voice> envelope and use the
        # rest. This is the ablated-pass safety net.
        speech = _VOICE_RE.sub("", text).strip()

    style = voice_m.group(1).strip() if voice_m else _DEFAULT_STYLE

    # Defensive truncation. OpenAI will accept long inputs but the UX
    # is awful and the cost climbs fast.
    if len(speech) > _TEXT_CAP:
        speech = speech[:_TEXT_CAP].rsplit(" ", 1)[0] + "…"
    if len(style) > _STYLE_CAP:
        style = style[:_STYLE_CAP]
    if not speech:
        # Worst case: nothing intelligible came out. Give the browser
        # something playable so the UX doesn't deadlock on a silent
        # request.
        speech = "(no audible response)"

    return speech, style


async def synthesize_mp3(
    *,
    text: str,
    style: str,
    voice: str,
) -> bytes:
    """One-shot TTS render. Returns WAV bytes (function name kept for
    back-compat — the original implementation used MP3).

    We deliberately do not stream the upstream response back chunked.
    Chat replies are short; the browser decodes the audio via
    `AudioContext.decodeAudioData` once the response body lands.

    Format note: we request WAV (raw PCM in a RIFF container) rather
    than MP3. Safari's `decodeAudioData` has been observed to decode
    OpenAI's gpt-4o-mini-tts MP3 stream into a buffer with the
    correct duration metadata but near-silent samples — the
    AudioContext registers as "playing" (tab icon turns on) but no
    audible sound reaches the speakers. WAV bypasses the codec
    entirely; decodeAudioData just reads the PCM samples directly.
    The download is larger (~10x) but TTS clips are short (<30s) so
    the wire-time cost is trivial.
    """
    key = settings.openai_api_key
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is empty — voice mode requires the key in .env"
        )

    payload = {
        "model": settings.tts_model,
        "input": text,
        "instructions": style,
        "voice": voice,
        "response_format": "wav",
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    timeout = aiohttp.ClientTimeout(total=60.0, connect=10.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            OPENAI_TTS_URL, json=payload, headers=headers,
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                logger.warning(
                    "OpenAI TTS returned %d for voice=%s: %s",
                    resp.status, voice, body[:500],
                )
                raise RuntimeError(
                    f"OpenAI TTS error {resp.status}: {body[:200]}"
                )
            return await resp.read()
