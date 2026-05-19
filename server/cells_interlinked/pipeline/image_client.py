"""Google Gemini image generation ("Nano Banana") for /chat imagery.

The chat path uses this when a turn has imagery enabled: each side
generates an image-prompt via a separate Gemma pass (ablated under the
same α the content was, for the ablated side); we send that prompt to
Gemini and persist the resulting PNG under data/chat_images/. The
server then exposes it via the /chat-images static mount.

`google-genai`'s `generate_content()` is synchronous (the SDK is built
on HTTPX but exposes a sync client). We run it on a thread executor so
the asyncio event loop driving the SSE stream isn't blocked while the
Gemini call is in flight (typically 4-10s for flash-image).
"""

from __future__ import annotations

import asyncio
import logging
from io import BytesIO
from pathlib import Path

from ..config import settings

logger = logging.getLogger(__name__)


# Borderline prompts (typical of the high-α ablated channel) trigger
# non-deterministic Nano Banana failures: the same prompt that fails
# on one call returns an image on the next. Empirically the API
# returns `parts=[(None, None)]` — a single part with no inline_data
# and no text — for ~40% of calls on some borderline prompts, even
# at temperature defaults. 5 retries puts the at-least-one-success
# probability above 99% for that level of flakiness; safety refusals
# are usually deterministic and will still fail after the full set.
_IMAGE_RETRIES = 5


def _generate_image_sync(prompt: str, output_path: Path, model: str, api_key: str) -> Path:
    """Blocking call into google-genai. Returns the saved PNG path.

    Failure modes we've observed:
      - Safety filter rejection — response carries explanatory text
        in place of image bytes. Deterministic; retry won't help.
      - `finish_reason=NO_IMAGE` — Nano Banana decided it couldn't
        make an image (typically from a too-sparse prompt) and
        returned a candidate with `content=None`. Sometimes flaky;
        retry sometimes succeeds.
      - `finish_reason=STOP` with parts that contain text but no
        inline_data — also flaky on borderline prompts. Retry.

    We let the model pick its own response modality (rather than
    forcing `IMAGE`) because forcing converts the chatty-text
    failure mode into the harder-to-recover NO_IMAGE mode.
    """
    from google import genai
    from PIL import Image

    client = genai.Client(api_key=api_key)
    last_err: Exception | None = None
    for attempt in range(_IMAGE_RETRIES + 1):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
        except Exception as exc:
            last_err = exc
            continue

        if not response.candidates:
            last_err = RuntimeError("Gemini returned no candidates")
            continue
        cand = response.candidates[0]
        finish = getattr(cand, "finish_reason", None)
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) if content else None

        if parts:
            for part in parts:
                if getattr(part, "inline_data", None) is not None:
                    image = Image.open(BytesIO(part.inline_data.data))
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(str(output_path), "PNG")
                    return output_path

            text_payload = ""
            for part in parts:
                if getattr(part, "text", None):
                    text_payload = part.text
                    break
            # parts=[(None, None)] is the empty-shell case we expect
            # to retry; parts with real text content but no image is
            # the safety-refusal case (still retried, but unlikely to
            # recover). Distinguish in the error message so logs
            # show which mode we exhausted on.
            kind = (
                "empty-shell parts (transient)"
                if not text_payload
                else f"text-only reply (likely refusal): {text_payload[:200]!r}"
            )
            last_err = RuntimeError(
                f"Gemini returned no image data (finish={finish}, {kind})"
            )
            continue

        last_err = RuntimeError(
            f"Gemini returned a candidate with no content (finish={finish})"
        )

    assert last_err is not None
    raise last_err


async def generate_image(
    *,
    prompt: str,
    output_path: Path,
    model: str | None = None,
) -> Path:
    """Async wrapper. Resolves to the absolute path of the saved PNG.

    The `prompt` is sent verbatim to Gemini. We deliberately do NOT
    wrap it with assistant-reply text or extra "generate an image"
    framing: an earlier iteration appended the reply as "atmosphere
    context" to ground sparse ablated prompts, and Nano Banana
    interpreted the quoted reply as text-to-render-into-the-image,
    inscribing literal sentences onto the output. The retry loop
    (`_IMAGE_RETRIES`) handles transient API flakes; the
    one-shot image-prompt template upstream produces substantive
    prompts on its own.
    """
    api_key = settings.google_api_key
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not configured — image gen unavailable")
    selected = model or settings.image_model
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, _generate_image_sync, prompt, output_path, selected, api_key,
    )


def image_path_for(session_id: str, turn_idx: int, side: str) -> Path:
    """Canonical on-disk path for a generated chat image."""
    return settings.image_dir / session_id / f"{turn_idx:04d}_{side}.png"


def image_url_for(session_id: str, turn_idx: int, side: str) -> str:
    """URL the browser uses to fetch the image, via the static mount."""
    return f"/chat-images/{session_id}/{turn_idx:04d}_{side}.png"
