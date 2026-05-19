"""HTTP route for the /chat voice mode.

The browser POSTs the parsed text + style from one channel along with
which side it is (raw or ablated). The server picks the appropriate
voice from settings, calls OpenAI's `gpt-4o-mini-tts`, and streams
the resulting MP3 bytes back. The key never leaves the server.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..config import settings
from ..pipeline.tts import synthesize_mp3

logger = logging.getLogger(__name__)
router = APIRouter()


class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    style: str = Field(default="", max_length=1500)
    side: str = Field(..., pattern="^(raw|ablated)$")


@router.post("/tts/speak")
async def speak(req: SpeakRequest) -> Response:
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY not configured — voice mode unavailable",
        )
    voice = (
        settings.tts_voice_raw
        if req.side == "raw"
        else settings.tts_voice_ablated
    )
    try:
        audio = await synthesize_mp3(
            text=req.text,
            style=req.style or "Speak in a calm, neutral tone.",
            voice=voice,
        )
    except RuntimeError as exc:
        logger.exception("TTS synthesis failed")
        raise HTTPException(status_code=502, detail=str(exc))
    return Response(
        content=audio,
        # WAV (PCM in RIFF) — Safari's decodeAudioData handles this
        # reliably; MP3 sometimes decoded to a buffer of near-silent
        # samples on Safari, which is the bug we worked around by
        # switching format in pipeline/tts.py.
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-store",
            # The /chat page is on a different port (3001) than the
            # API (8000). The default CORS middleware allows the GET
            # but the browser still wants explicit access for the
            # blob to be playable in some setups; this is belt-and-
            # suspenders since we already configure CORSMiddleware.
            "Access-Control-Expose-Headers": "Content-Length, Content-Type",
        },
    )
