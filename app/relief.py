"""
AI relief generation: re-render a source image (an upload, or an earlier
result) as a sculpted silver relief, ready for the heightmap pipeline.

Providers:
- FalRelief (preferred): instruction-following image-editing models on fal.ai
  -- FLUX.1 Kontext [pro] for single images and the cheaper Kontext [dev] for
  the 4-variation requests by default (config.FAL_MODEL / FAL_MODEL_VARIATIONS;
  Qwen Image Edit and FLUX.2 edit work too). The prompt is an instruction
  ("Transform this image into ..."), with faithfulness spelled out in words.
- StabilityRelief: Stable Diffusion via Stability AI's Structure Control
  endpoint (POST /v2beta/stable-image/control/structure). It keeps the source
  image's structure and re-renders it from the prompt; `control_strength`
  (the panel's fidelity slider) sets how closely. One image per call.
- MockRelief: a free, offline stand-in (no API key) that turns the source
  into a grey sculpted-looking image, so the feature can be used and tested
  without an account. Clearly labelled; costs nothing.

Every generated image is logged to config.USAGE_LOG (JSON lines) with its
dollar cost (and Stability credits), for the usage counter in the panel.
"""
from __future__ import annotations

import base64
import io
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from app import config

STRUCTURE_PATH = "/v2beta/stable-image/control/structure"
BALANCE_PATH = "/v1/user/balance"
_MIN_SIDE, _MAX_PIXELS, _MAX_ASPECT = 64, 9_437_184, 2.5


class ReliefError(RuntimeError):
    """Generation failed (bad key, rejected prompt, network, ...)."""


@dataclass
class ReliefImage:
    rgb: np.ndarray
    seed: int
    credits: float       # Stability credits (0 for other providers)
    usd: float = 0.0
    model: str = ""


def _fit_for_api(rgb: np.ndarray) -> np.ndarray:
    """Keep the source within the API's size and aspect limits (crop to 2.5:1)."""
    h, w = rgb.shape[:2]
    if w / h > _MAX_ASPECT:
        nw = int(h * _MAX_ASPECT)
        rgb = rgb[:, (w - nw) // 2:(w - nw) // 2 + nw]
    elif h / w > _MAX_ASPECT:
        nh = int(w * _MAX_ASPECT)
        rgb = rgb[(h - nh) // 2:(h - nh) // 2 + nh]
    h, w = rgb.shape[:2]
    if w * h > _MAX_PIXELS:
        s = (_MAX_PIXELS / (w * h)) ** 0.5 * 0.99
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    if min(rgb.shape[:2]) < _MIN_SIDE:
        s = _MIN_SIDE / min(rgb.shape[:2])
        rgb = cv2.resize(rgb, (int(np.ceil(rgb.shape[1] * s)), int(np.ceil(rgb.shape[0] * s))))
    return rgb


class StabilityRelief:
    name = "stability"
    instruction = False  # a descriptive prompt; faithfulness is control_strength

    def __init__(self):
        if not config.STABILITY_API_KEY:
            raise ReliefError("STABILITY_API_KEY is not set. Add it to your .env file to generate with Stable Diffusion.")
        import httpx

        self._http = httpx.Client(timeout=120)

    def price_usd(self, variation: bool = False) -> float:
        return config.STABILITY_CREDITS_PER_IMAGE * config.STABILITY_USD_PER_CREDIT

    def generate(self, rgb: np.ndarray, prompt: str, negative: str, fidelity: float, seed: int,
                 variation: bool = False) -> ReliefImage:
        buf = io.BytesIO()
        Image.fromarray(_fit_for_api(rgb)).save(buf, format="PNG")
        try:
            res = self._http.post(
                config.STABILITY_API_BASE + STRUCTURE_PATH,
                headers={"Authorization": f"Bearer {config.STABILITY_API_KEY}", "Accept": "image/*"},
                files={"image": ("source.png", buf.getvalue(), "image/png")},
                data={"prompt": prompt[:10000], "negative_prompt": negative[:10000],
                      "control_strength": f"{float(np.clip(fidelity, 0, 1)):.2f}",
                      "seed": str(seed), "output_format": "png"},
            )
        except Exception as err:
            raise ReliefError(f"Couldn't reach Stability AI: {err}") from err
        if res.status_code != 200:
            try:
                detail = "; ".join(res.json().get("errors", [])) or res.text
            except ValueError:
                detail = res.text
            raise ReliefError(f"Stability AI error {res.status_code}: {detail[:300]}")
        if res.headers.get("finish-reason", "SUCCESS") == "CONTENT_FILTERED":
            raise ReliefError("Stability AI's content filter blocked this image. Try a different request.")
        out = np.array(Image.open(io.BytesIO(res.content)).convert("RGB"))
        return ReliefImage(out, int(res.headers.get("seed", seed)), config.STABILITY_CREDITS_PER_IMAGE,
                           self.price_usd(), "stable-image-control-structure")

    def balance(self) -> float | None:
        try:
            res = self._http.get(config.STABILITY_API_BASE + BALANCE_PATH,
                                 headers={"Authorization": f"Bearer {config.STABILITY_API_KEY}"})
            return float(res.json()["credits"]) if res.status_code == 200 else None
        except Exception:
            return None


class MockRelief:
    """Offline stand-in: a grey, embossed, smoothed version of the source,
    varied a little per seed. Not a real model -- just so the flow works."""
    name = "mock"
    instruction = False

    def price_usd(self, variation: bool = False) -> float:
        return 0.0

    def generate(self, rgb: np.ndarray, prompt: str, negative: str, fidelity: float, seed: int,
                 variation: bool = False) -> ReliefImage:
        rng = np.random.default_rng(seed)
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        g = cv2.bilateralFilter(g, 0, 0.15, 6)
        k = 3 + 2 * int(rng.integers(0, 3))
        gy, gx = np.gradient(cv2.GaussianBlur(g, (k, k), 0))
        a = rng.uniform(0, 2 * np.pi)
        shade = 0.55 + 0.4 * g + 6.0 * (np.cos(a) * gx + np.sin(a) * gy)
        if "plain, flat" in prompt:  # "No background": fade the edges to a flat field
            h, w = g.shape
            yy, xx = np.mgrid[:h, :w]
            r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2))
            shade = shade * np.clip(1.4 - r, 0, 1) + 0.25 * (1 - np.clip(1.4 - r, 0, 1))
        out = (np.clip(shade, 0, 1) * 255).astype(np.uint8)
        time.sleep(0.2)
        return ReliefImage(np.dstack([out] * 3), seed, 0.0, 0.0, "mock")

    def balance(self) -> float | None:
        return None


# --- fal.ai ---------------------------------------------------------------------------

FAL_RUN = "https://fal.run/"
FAL_SOURCE_MAX_MP = 1.0  # the source is sent at up to ~1 MP (the models work at about that size)


def fal_price_usd(model: str, mp: float = 1.0) -> float:
    """fal.ai's list price for one image of `mp` megapixels (output; the
    source is sent at about 1 MP), as published on each model's page."""
    mp_up = max(1, math.ceil(mp - 1e-9))
    if model.endswith("flux-pro/kontext") or "flux-pro/kontext/max" in model:
        return 0.08 if "max" in model else 0.04          # per image
    if model.endswith("flux-kontext/dev"):
        return 0.025 * mp                                  # per megapixel
    if "qwen-image-edit" in model:
        return 0.03 * mp                                   # per megapixel
    if model.endswith("flux-2/edit"):
        return 0.012 * (mp + 1.0)                          # per megapixel, input + output
    if model.endswith("flux-2-pro/edit"):
        return 0.03 + 0.015 * (mp_up - 1)                  # first MP $0.03, each extra MP $0.015
    return 0.04                                            # unknown model: a cautious estimate


class FalRelief:
    """Image editing on fal.ai (synchronous endpoint, one image per call)."""
    name = "fal"
    instruction = True

    def __init__(self):
        if not config.FAL_KEY:
            raise ReliefError("FAL_KEY is not set. Add it to your .env file to generate with fal.ai.")
        import httpx

        self._http = httpx.Client(timeout=180, follow_redirects=True)

    def model_for(self, variation: bool) -> str:
        return config.FAL_MODEL_VARIATIONS if variation else config.FAL_MODEL

    def price_usd(self, variation: bool = False) -> float:
        return fal_price_usd(self.model_for(variation), 1.0)

    @staticmethod
    def _data_uri(rgb: np.ndarray) -> tuple[str, int, int]:
        h, w = rgb.shape[:2]
        s = min(1.0, (FAL_SOURCE_MAX_MP * 1e6 / (w * h)) ** 0.5)
        if s < 1.0:
            rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=92)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), rgb.shape[1], rgb.shape[0]

    def _payload(self, model: str, image: str, w: int, h: int, prompt: str, negative: str, seed: int) -> dict:
        body = {"prompt": prompt, "seed": seed, "num_images": 1, "output_format": "png"}
        if "qwen-image-edit-plus" in model or "flux-2" in model:
            body["image_urls"] = [image]
        else:
            body["image_url"] = image
        if "qwen-image-edit" in model:
            body["negative_prompt"] = negative
        if "flux-pro/kontext" in model:
            body["safety_tolerance"] = "2"
        if "qwen" in model or "flux-2" in model:
            body["image_size"] = {"width": w - w % 16, "height": h - h % 16}  # keep the source's shape
        return body

    def generate(self, rgb: np.ndarray, prompt: str, negative: str, fidelity: float, seed: int,
                 variation: bool = False) -> ReliefImage:
        model = self.model_for(variation)
        image, w, h = self._data_uri(rgb)
        try:
            res = self._http.post(FAL_RUN + model, headers={"Authorization": f"Key {config.FAL_KEY}"},
                                  json=self._payload(model, image, w, h, prompt, negative, seed))
        except Exception as err:
            raise ReliefError(f"Couldn't reach fal.ai: {err}") from err
        if res.status_code != 200:
            try:
                detail = res.json().get("detail", res.text)
                detail = detail if isinstance(detail, str) else json.dumps(detail)
            except ValueError:
                detail = res.text
            raise ReliefError(f"fal.ai error {res.status_code} ({model}): {detail[:300]}")
        data = res.json()
        if any(data.get("has_nsfw_concepts") or []):
            raise ReliefError("fal.ai's safety checker blocked this image. Try a different request.")
        try:
            url = data["images"][0]["url"]
        except (KeyError, IndexError, TypeError) as err:
            raise ReliefError(f"fal.ai returned no image ({model})") from err
        if url.startswith("data:"):
            raw = base64.b64decode(url.split(",", 1)[1])
        else:
            img_res = self._http.get(url)
            if img_res.status_code != 200:
                raise ReliefError(f"Couldn't download the result from fal.ai ({img_res.status_code})")
            raw = img_res.content
        out = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
        usd = fal_price_usd(model, out.shape[0] * out.shape[1] / 1e6)
        return ReliefImage(out, int(data.get("seed", seed)), 0.0, round(usd, 4), model)

    def balance(self) -> float | None:
        return None  # fal.ai has no public balance endpoint; see fal.ai/dashboard/billing


def get_relief_provider():
    if config.RELIEF_PROVIDER == "fal":
        return FalRelief()
    if config.RELIEF_PROVIDER == "stability":
        return StabilityRelief()
    return MockRelief()


def generate_many(provider, rgb: np.ndarray, prompt: str, negative: str, fidelity: float, n: int) -> list[ReliefImage]:
    """n images with different random seeds, requested in parallel. More than
    one is a "variations" request (fal: the cheaper variations model)."""
    seeds = [int(s) for s in np.random.default_rng().integers(0, 2_147_483_647, n)]
    with ThreadPoolExecutor(max_workers=min(n, 4)) as pool:
        return list(pool.map(lambda s: provider.generate(rgb, prompt, negative, fidelity, s, variation=n > 1), seeds))


# --- usage -------------------------------------------------------------------------

_usage_lock = threading.Lock()
_balance_cache: dict = {"at": 0.0, "credits": None}


def log_usage(provider: str, img: ReliefImage, candidate_id: str, request_id: str) -> None:
    rec = {"ts": time.time(), "provider": provider, "model": img.model, "credits": img.credits,
           "usd": round(img.usd, 4), "candidate_id": candidate_id, "request_id": request_id}
    with _usage_lock:
        config.USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(config.USAGE_LOG, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    _balance_cache["at"] = 0.0  # refresh the balance next time


def usage_summary(provider=None) -> dict:
    """Totals so far (all time and this session's day), plus the account balance if known."""
    images = credits = usd = 0.0
    today_images = today_credits = today_usd = 0.0
    day_start = time.time() - time.time() % 86400
    if config.USAGE_LOG.exists():
        for line in config.USAGE_LOG.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            images += 1
            credits += r.get("credits", 0)
            usd += r.get("usd", 0)
            if r.get("ts", 0) >= day_start:
                today_images += 1
                today_credits += r.get("credits", 0)
                today_usd += r.get("usd", 0)
    balance = None
    if provider is not None and time.time() - _balance_cache["at"] > 60:
        _balance_cache["credits"] = provider.balance()
        _balance_cache["at"] = time.time()
    if provider is not None:
        balance = _balance_cache["credits"]
    return {
        "provider": config.RELIEF_PROVIDER,
        "images": int(images), "credits": round(credits, 2), "usd": round(usd, 3),
        "today": {"images": int(today_images), "credits": round(today_credits, 2), "usd": round(today_usd, 3)},
        "balance_credits": balance,
    }
