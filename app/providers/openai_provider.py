"""
OpenAIProvider: real text->image generation via OpenAI's GPT Image models
(gpt-image-2.5-flare by default; see config.OPENAI_IMAGE_MODEL).

Requires OPENAI_API_KEY in the environment (.env). These models only accept
a fixed set of sizes (no arbitrary width/height), so we pick whichever
supported size best matches the requested aspect ratio and let the existing
cover-fit-resize step (in app/main.py) crop it to the exact face aspect
ratio afterwards.
"""
from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np
from PIL import Image

from app import config
from app.providers.base import ImageProvider

# (width, height) pairs OpenAI's image API currently accepts.
_SUPPORTED_SIZES = [(1024, 1024), (1536, 1024), (1024, 1536)]


class OpenAIProvider(ImageProvider):
    name = "openai"

    def __init__(self):
        if not config.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file to use IMAGE_PROVIDER=openai."
            )
        from openai import OpenAI

        self._client = OpenAI(api_key=config.OPENAI_API_KEY)

    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        n: int = 1,
        seed: Optional[int] = None,
    ) -> list[np.ndarray]:
        # OpenAI's Images API has no seed parameter, so `seed` is accepted
        # for interface compatibility but has no effect here. The app
        # (app/main.py) always calls this with n=1 to keep cost predictable.
        size_w, size_h = _closest_supported_size(width, height)

        try:
            response = self._client.images.generate(
                model=config.OPENAI_IMAGE_MODEL,
                prompt=prompt,
                size=f"{size_w}x{size_h}",
                quality=config.OPENAI_IMAGE_QUALITY,
                n=n,
            )
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI image generation failed: {exc}. If this is a permissions error, "
                "note OpenAI's image models require your organization to be verified."
            ) from exc

        images = []
        for item in response.data:
            raw = base64.b64decode(item.b64_json)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            images.append(np.array(img))
        return images


def _closest_supported_size(target_w: int, target_h: int) -> tuple[int, int]:
    target_log_aspect = np.log(target_w / target_h)
    return min(
        _SUPPORTED_SIZES,
        key=lambda wh: abs(np.log(wh[0] / wh[1]) - target_log_aspect),
    )
