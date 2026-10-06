from __future__ import annotations

import io
import uuid
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from app import background, config, enhance, prompts, relief, relief_prompts, storage, textures
from app.imaging import cap_max_dimension, cover_fit_resize
from app.processing import ProcessParams, process_image
from app.providers import get_provider

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_UPLOAD_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}

app = FastAPI(title="Ring Face Relief Designer")


# --- request/response models ---------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500)
    preset: str = Field(default=prompts.DEFAULT_PRESET)
    seed: Optional[int] = None


class CandidateOut(BaseModel):
    candidate_id: str
    thumbnail_url: str


class GenerateResponse(BaseModel):
    prompt: str
    preset: str
    seed: Optional[int]
    candidates: list[CandidateOut]
    warnings: list[str] = []


class ReliefRequest(BaseModel):
    source_id: str                                   # the selected candidate; results always come from its original upload
    style: str = relief_prompts.DEFAULT_STYLE
    image_type: str = relief_prompts.DEFAULT_IMAGE_TYPE  # portrait, animal, landscape, painting, ...
    additions: list[str] = []                        # ids from relief_prompts.ADDITIONS, in the order picked
    text: str = Field(default="", max_length=500)    # the user's own request; if given, 4 variations are made
    fidelity: float = Field(default=config.RELIEF_FIDELITY_DEFAULT, ge=0.0, le=1.0)
    model: str | None = None                         # fal: one of config.FAL_MODEL_CHOICES, for comparing models


class ReliefResponse(BaseModel):
    request_id: str
    prompt: str
    candidates: list[CandidateOut]
    usage: dict


class ProcessRequest(BaseModel):
    candidate_id: str
    invert: bool = False
    gamma: float = 1.0
    contrast: float = 1.0
    blur_mm: float = 0.0
    levels: int = 0
    min_feature_mm: float = config.MIN_FEATURE_MM
    relief_height_mm: float = 0.25
    # Mirror the image left-right / top-bottom before it becomes the heightmap.
    flip_h: bool = False
    flip_v: bool = False
    # Which part of the full source image maps onto the face. zoom=1,
    # offset=(0,0) is the default centered cover-fit crop; the frontend
    # cropper lets a user adjust these without re-calling the image model.
    crop_zoom: float = Field(default=1.0, ge=config.CROP_ZOOM_MIN, le=config.CROP_ZOOM_MAX)
    crop_offset_x: float = Field(default=0.0, ge=-1.0, le=1.0)
    crop_offset_y: float = Field(default=0.0, ge=-1.0, le=1.0)
    # Which design ring the heightmap is for ("Shape/Size", see relief.json;
    # None = the default), and for a "tilt" ring the angle that sets its
    # design area (the viewer's slider; None = its default).
    relief_ring: Optional[str] = None
    face_tilt_deg: Optional[float] = None
    # For a "recess" ring: how far its ridge's inner wall is slid outward (mm),
    # which widens the floor the design fills (the viewer's "ridge wall" slider).
    ridge_shift_mm: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # Keep only the image's main subject: the background becomes a flat field
    # and the subject a raised plateau (see app/background.py).
    remove_background: bool = False
    # Image enhancement (see app/processing.py and app/enhance.py). Ring-agnostic:
    # every design ring gets these.
    height_source: str = Field(default="brightness", pattern="^(brightness|depth)$")  # depth: AI 3D shape
    upscale: bool = False            # AI 2x super-resolution of the source (cached per candidate)
    denoise: bool = False            # edge-preserving cleanup of the source
    bas_relief: float = Field(default=0.0, ge=0.0, le=1.0)
    depth_detail: float = Field(default=0.0, ge=0.0, le=1.0)
    outline_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    hatch_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    hatch_spacing_mm: float = Field(default=0.6, ge=0.2, le=2.0)
    hatch_angle_deg: float = Field(default=45.0, ge=0.0, le=180.0)
    smooth_mm: float = Field(default=0.0, ge=0.0, le=0.3)


class ReportOut(BaseModel):
    coverage_percent: float
    relief_volume_mm3: float
    estimated_weight_g: float
    warnings: list[str]
    passed: bool


class ProcessResponse(BaseModel):
    design_id: str
    preview_url: str
    heightmap_url: str
    params_url: str
    report: ReportOut


# --- endpoints ---------------------------------------------------

@app.get("/api/config")
def get_config():
    return {
        "face_width_mm": config.FACE_WIDTH_MM,
        "face_height_mm": config.FACE_HEIGHT_MM,
        "px_per_mm": config.PX_PER_MM,
        "heightmap_width_px": config.HEIGHTMAP_WIDTH_PX,
        "heightmap_height_px": config.HEIGHTMAP_HEIGHT_PX,
        "relief_max_mm": config.RELIEF_MAX_MM,
        "silver_alloy": config.SILVER_ALLOY,
        "silver_density_g_cm3": config.SILVER_DENSITY_G_CM3,
        "edge_margin_mm": config.EDGE_MARGIN_MM,
        "edge_feather_mm": config.EDGE_FEATHER_MM,
        "min_feature_mm": config.MIN_FEATURE_MM,
        "ring_diameter_mm": config.RING_DIAMETER_MM,
        "ring_band_thickness_mm": config.RING_BAND_THICKNESS_MM,
        "crop_zoom_min": config.CROP_ZOOM_MIN,
        "crop_zoom_max": config.CROP_ZOOM_MAX,
        "presets": {
            pid: {"label": p.label, "levels": p.levels, "blur_mm": p.blur_mm}
            for pid, p in prompts.PRESETS.items()
        },
        "default_preset": prompts.DEFAULT_PRESET,
        "image_provider": config.IMAGE_PROVIDER,
    }


@app.post("/api/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if req.preset not in prompts.PRESETS:
        raise HTTPException(400, f"Unknown preset '{req.preset}'. Options: {list(prompts.PRESETS)}")

    try:
        provider = get_provider()
    except RuntimeError as exc:
        raise HTTPException(400, str(exc))

    full_prompt = prompts.build_prompt(req.prompt, req.preset)

    # Generate at the face aspect ratio (cover-fit to exact size after).
    gen_long_edge = config.CANDIDATE_GEN_LONG_EDGE_PX
    if config.FACE_WIDTH_MM >= config.FACE_HEIGHT_MM:
        gen_w = gen_long_edge
        gen_h = int(round(gen_long_edge * config.FACE_HEIGHT_MM / config.FACE_WIDTH_MM))
    else:
        gen_h = gen_long_edge
        gen_w = int(round(gen_long_edge * config.FACE_WIDTH_MM / config.FACE_HEIGHT_MM))

    try:
        # Always request exactly one image per call -- each call is a real,
        # billed API request, and the UI lets a user add more one at a time
        # (via the same "Generate" button) rather than forcing a batch.
        raw_images = provider.generate(full_prompt, gen_w, gen_h, 1, req.seed)
    except RuntimeError as exc:
        raise HTTPException(502, str(exc))

    warnings = []
    if prompts.mentions_text(req.prompt):
        warnings.append(
            "Your prompt mentions text/lettering. This MVP's image model cannot render "
            "legible text on the ring face — it will be skipped or come out as abstract shapes."
        )

    candidates = []
    for raw_img in raw_images:
        # Keep the full image (not pre-cropped to the face aspect ratio) so
        # the user can pan/zoom which part of it lands on the face later,
        # via /api/process's crop_* params, without a re-generation call.
        full_img = cap_max_dimension(raw_img, config.MAX_FULL_IMAGE_DIM_PX)
        design_id, d = storage.new_design_dir()
        storage.save_rgb_png(d / "candidate_full.png", full_img)
        thumb = storage.make_thumbnail(full_img)
        storage.save_rgb_png(d / "thumbnail.png", thumb)
        storage.write_json(d / "meta.json", {
            "design_id": design_id,
            "prompt": req.prompt,
            "preset": req.preset,
            "full_prompt": full_prompt,
            "seed": req.seed,
            "provider": provider.name,
            "model": getattr(provider, "name", "unknown"),
            "model_slug": config.OPENAI_IMAGE_MODEL if provider.name == "openai" else "mock",
        })
        candidates.append(CandidateOut(
            candidate_id=design_id,
            thumbnail_url=f"/api/designs/{design_id}/thumbnail.png",
        ))

    return GenerateResponse(prompt=req.prompt, preset=req.preset, seed=req.seed, candidates=candidates, warnings=warnings)


@app.get("/api/relief/options")
def relief_options():
    """Styles and additions for the "Design with AI" panel, plus the provider and its prices."""
    try:
        provider = relief.get_relief_provider()
        price_single, price_variation = provider.price_usd(False), provider.price_usd(True)
        models = {"single": provider.model_for(False), "variations": provider.model_for(True)} \
            if hasattr(provider, "model_for") else None
    except relief.ReliefError:
        price_single = price_variation = 0.0
        models = None
    # the model picker (fal only); the configured single-image model is
    # preselected when it's one of the choices
    choices = [{"id": m, "label": label, "price_usd": relief.fal_price_usd(m, 1.0)}
               for m, label in config.FAL_MODEL_CHOICES] if config.RELIEF_PROVIDER == "fal" else []
    default_model = next((c["id"] for c in choices if c["id"] == config.FAL_MODEL), None)
    return {**relief_prompts.options(), "provider": config.RELIEF_PROVIDER, "models": models,
            "model_choices": choices, "default_model": default_model,
            "price_usd_single": price_single, "price_usd_variation": price_variation,
            "images_with_text": config.RELIEF_IMAGES_WITH_TEXT, "images_default": config.RELIEF_IMAGES_DEFAULT,
            "fidelity_default": config.RELIEF_FIDELITY_DEFAULT}


@app.post("/api/relief", response_model=ReliefResponse)
def generate_relief(req: ReliefRequest):
    """
    Re-render a candidate as a sculpted silver relief with an image model
    (fal.ai editing models, Stability Structure Control, or the offline
    stand-in -- see app/relief.py), from the locked base prompt + style +
    additions + the user's text. A typed
    request makes RELIEF_IMAGES_WITH_TEXT variations; presets alone make one.
    Each result is stored as a new candidate, so every slider and the 3D
    preview work on it; its meta records how it was made and what it cost.
    """
    if not (storage.design_dir(req.source_id) / "candidate_full.png").exists():
        raise HTTPException(404, f"Unknown candidate '{req.source_id}'")
    if req.style not in relief_prompts.STYLES:
        raise HTTPException(400, f"Unknown style '{req.style}'")
    if req.image_type not in relief_prompts.IMAGE_TYPES:
        raise HTTPException(400, f"Unknown image type '{req.image_type}'")
    if req.model and req.model not in dict(config.FAL_MODEL_CHOICES):
        raise HTTPException(400, f"Unknown model '{req.model}'")
    # Always work from the original image: re-generating from a result drifts
    # further from the original each time.
    origin_id = _original_of(req.source_id)
    src = storage.design_dir(origin_id) / "candidate_full.png"
    rgb = cv2.cvtColor(cv2.imread(str(src), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    n = config.RELIEF_IMAGES_WITH_TEXT if req.text.strip() else config.RELIEF_IMAGES_DEFAULT
    try:
        provider = relief.get_relief_provider()
        prompt, negative = relief_prompts.build_relief_prompt(
            req.style, req.additions, req.text, instruction=provider.instruction, fidelity=req.fidelity,
            image_type=req.image_type)
        images = relief.generate_many(provider, rgb, prompt, negative, req.fidelity, n, model=req.model)
    except relief.ReliefError as err:
        raise HTTPException(502, str(err))

    request_id = uuid.uuid4().hex
    out = []
    h0, w0 = rgb.shape[:2]
    bg_choice = relief_prompts.background_choice(req.additions)
    for img in images:
        # the original's exact aspect ratio (models snap to their own sizes),
        # at the result's own resolution
        h, w = img.rgb.shape[:2]
        scale = min(w / w0, h / h0)
        same_shape = cover_fit_resize(img.rgb, max(1, round(w0 * scale)), max(1, round(h0 * scale)))
        full = cap_max_dimension(same_shape, config.MAX_FULL_IMAGE_DIM_PX)
        design_id, d = storage.new_design_dir()
        if bg_choice:
            # The model drew the subject on a plain background: cut it out and
            # build the background here -- flat, shadow-free, with the chosen
            # texture (raised slightly in the heightmap, see /api/process).
            storage.save_rgb_png(d / "generated.png", full)  # the model's own image, before compositing
            try:
                mask = background.solid_subject_mask(full)
            except background.BackgroundRemovalUnavailable as err:
                raise HTTPException(503, str(err))
            cv2.imwrite(str(d / "subject_mask.png"), mask)
            m = (mask.astype(np.float32) / 255.0)[..., None]
            flat = textures.flat_preview(bg_choice.texture, full.shape[1], full.shape[0])
            full = np.clip(full * m + flat * (1 - m), 0, 255).astype(np.uint8)
        storage.save_rgb_png(d / "candidate_full.png", full)
        storage.save_rgb_png(d / "thumbnail.png", storage.make_thumbnail(full))
        storage.write_json(d / "meta.json", {
            "design_id": design_id, "source": "relief", "parent": origin_id, "request_id": request_id,
            "prompt": req.text, "style": req.style, "image_type": req.image_type, "additions": req.additions,
            "fidelity": req.fidelity,
            "background": {"choice": bg_choice.id, "texture": bg_choice.texture} if bg_choice else None,
            "full_prompt": prompt, "negative_prompt": negative, "seed": img.seed,
            "provider": provider.name, "model": img.model,
            "credits": img.credits, "usd": img.usd, "preset": None,
        })
        relief.log_usage(provider.name, img, design_id, request_id)
        out.append(CandidateOut(candidate_id=design_id, thumbnail_url=f"/api/designs/{design_id}/thumbnail.png"))
    return ReliefResponse(request_id=request_id, prompt=prompt, candidates=out, usage=relief.usage_summary(provider))


def _original_of(design_id: str) -> str:
    """The upload (or text-generated image) a candidate ultimately came from:
    follows AI results' `parent` links back to the start."""
    seen = set()
    while design_id not in seen:
        seen.add(design_id)
        try:
            meta = storage.read_json(storage.design_dir(design_id) / "meta.json")
        except (OSError, ValueError, HTTPException):
            break
        parent = meta.get("parent") if meta.get("source") == "relief" else None
        if not parent or not (storage.design_dir(parent) / "candidate_full.png").exists():
            break
        design_id = parent
    return design_id


@app.get("/api/usage")
def get_usage():
    """AI generation used so far: images, credits and dollars (all time and today), plus the account balance."""
    try:
        provider = relief.get_relief_provider()
    except relief.ReliefError:
        provider = None
    return relief.usage_summary(provider)


@app.post("/api/upload", response_model=CandidateOut)
async def upload_photo(file: UploadFile = File(...)):
    """
    Add a user-supplied photo as a candidate, bypassing the image model
    entirely. It flows through the same cover-fit + /api/process pipeline
    as a generated candidate.
    """
    if file.content_type not in ALLOWED_UPLOAD_CONTENT_TYPES:
        raise HTTPException(400, "Unsupported image type. Use PNG, JPEG, or WEBP.")

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, "Image too large (max 15MB).")

    try:
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image file.")

    rgb = np.array(pil_img)
    full_img = cap_max_dimension(rgb, config.MAX_FULL_IMAGE_DIM_PX)

    design_id, d = storage.new_design_dir()
    storage.save_rgb_png(d / "candidate_full.png", full_img)
    thumb = storage.make_thumbnail(full_img)
    storage.save_rgb_png(d / "thumbnail.png", thumb)
    storage.write_json(d / "meta.json", {
        "design_id": design_id,
        "prompt": "",
        "preset": None,
        "full_prompt": None,
        "seed": None,
        "provider": "upload",
        "model": "upload",
        "model_slug": None,
        "source": "upload",
        "source_filename": file.filename,
    })

    return CandidateOut(candidate_id=design_id, thumbnail_url=f"/api/designs/{design_id}/thumbnail.png")


@app.post("/api/process", response_model=ProcessResponse)
def process(req: ProcessRequest):
    d = storage.design_dir(req.candidate_id)
    candidate_path = d / "candidate_full.png"
    if not candidate_path.exists():
        raise HTTPException(404, f"Unknown candidate_id '{req.candidate_id}'")

    meta = storage.read_json(d / "meta.json")

    bgr = cv2.imread(str(candidate_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(500, "Failed to read cached candidate image")
    full_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Crop/zoom/pan happens here, every call, against the cached full image
    # -- never against the image model.
    face_w, face_h, face_outline = config.face_geometry(req.relief_ring, req.face_tilt_deg, req.ridge_shift_mm)
    hm_w, hm_h = int(round(face_w * config.PX_PER_MM)), int(round(face_h * config.PX_PER_MM))
    try:
        source_rgb = enhance.cached_upscaled(d, full_rgb) if req.upscale else full_rgb
        depth_full = enhance.cached_depth(d, full_rgb) if req.height_source == "depth" else None
    except enhance.ModelUnavailable as err:
        raise HTTPException(503, str(err))
    crop = dict(zoom=req.crop_zoom, offset_x=req.crop_offset_x, offset_y=req.crop_offset_y)
    depth = None if depth_full is None else cover_fit_resize(depth_full, hm_w, hm_h, **crop)
    rgb = cover_fit_resize(
        source_rgb,
        hm_w,
        hm_h,
        zoom=req.crop_zoom,
        offset_x=req.crop_offset_x,
        offset_y=req.crop_offset_y,
    )

    # AI designs made with a background option have their subject mask saved
    # and a background texture to lay behind it (flat, raised slightly)
    bg_meta = meta.get("background") or {}
    mask = None
    if req.remove_background or bg_meta:
        try:
            mask_full = background.cached_subject_mask(d, full_rgb)
        except background.BackgroundRemovalUnavailable as err:
            raise HTTPException(503, str(err))
        # the same crop/zoom/pan as the image
        mask = cover_fit_resize(mask_full, hm_w, hm_h, zoom=req.crop_zoom,
                                offset_x=req.crop_offset_x, offset_y=req.crop_offset_y)

    if req.flip_h:
        rgb = np.ascontiguousarray(rgb[:, ::-1])
        mask = None if mask is None else np.ascontiguousarray(mask[:, ::-1])
        depth = None if depth is None else np.ascontiguousarray(depth[:, ::-1])
    if req.flip_v:
        rgb = np.ascontiguousarray(rgb[::-1])
        mask = None if mask is None else np.ascontiguousarray(mask[::-1])
        depth = None if depth is None else np.ascontiguousarray(depth[::-1])

    params = ProcessParams(
        invert=req.invert,
        gamma=req.gamma,
        contrast=req.contrast,
        blur_mm=req.blur_mm,
        levels=req.levels,
        min_feature_mm=req.min_feature_mm,
        relief_height_mm=req.relief_height_mm,
        background_texture=bg_meta.get("texture"),
        denoise=req.denoise,
        bas_relief=req.bas_relief,
        depth_detail=req.depth_detail,
        outline_strength=req.outline_strength,
        hatch_strength=req.hatch_strength,
        hatch_spacing_mm=req.hatch_spacing_mm,
        hatch_angle_deg=req.hatch_angle_deg,
        smooth_mm=req.smooth_mm,
    )

    contains_text_request = prompts.mentions_text(meta.get("prompt", ""))
    heightmap_16bit, preview_8bit, report = process_image(
        rgb, params, contains_text_request, face_width_mm=face_w, face_height_mm=face_h, face_outline_mm=face_outline,
        subject_mask=None if mask is None else mask.astype(np.float32) / 255.0,
        depth=depth,
    )

    storage.save_gray16_png(d / "heightmap.png", heightmap_16bit)
    storage.save_gray8_png(d / "preview.png", preview_8bit)

    params_json = {
        **meta,
        "processing": {
            "invert": params.invert,
            "flip_h": req.flip_h,
            "flip_v": req.flip_v,
            "gamma": params.gamma,
            "contrast": params.contrast,
            "blur_mm": params.blur_mm,
            "levels": params.levels,
            "min_feature_mm": params.min_feature_mm,
            "relief_height_mm": params.relief_height_mm,
            "edge_margin_mm": params.edge_margin_mm,
            "edge_feather_mm": params.edge_feather_mm,
            "relief_max_mm": config.RELIEF_MAX_MM,
            "px_per_mm": config.PX_PER_MM,
            "face_width_mm": face_w,
            "face_height_mm": face_h,
            "relief_ring": req.relief_ring or config.DEFAULT_RELIEF_RING,
            "face_tilt_deg": req.face_tilt_deg,
            "ridge_shift_mm": req.ridge_shift_mm,
            "remove_background": req.remove_background,
            "height_source": req.height_source,
            "upscale": req.upscale,
            "denoise": req.denoise,
            "bas_relief": req.bas_relief,
            "depth_detail": req.depth_detail,
            "outline_strength": req.outline_strength,
            "hatch_strength": req.hatch_strength,
            "hatch_spacing_mm": req.hatch_spacing_mm,
            "hatch_angle_deg": req.hatch_angle_deg,
            "smooth_mm": req.smooth_mm,
            "crop_zoom": req.crop_zoom,
            "crop_offset_x": req.crop_offset_x,
            "crop_offset_y": req.crop_offset_y,
        },
        "report": {
            "coverage_percent": report.coverage_percent,
            "relief_volume_mm3": report.relief_volume_mm3,
            "estimated_weight_g": report.estimated_weight_g,
            "warnings": report.warnings,
            "passed": report.passed,
        },
    }
    storage.write_json(d / "params.json", params_json)

    return ProcessResponse(
        design_id=req.candidate_id,
        preview_url=f"/api/designs/{req.candidate_id}/preview.png",
        heightmap_url=f"/api/designs/{req.candidate_id}/heightmap.png",
        params_url=f"/api/designs/{req.candidate_id}/params.json",
        report=ReportOut(
            coverage_percent=report.coverage_percent,
            relief_volume_mm3=report.relief_volume_mm3,
            estimated_weight_g=report.estimated_weight_g,
            warnings=report.warnings,
            passed=report.passed,
        ),
    )


@app.get("/api/designs/{design_id}/thumbnail.png")
def get_thumbnail(design_id: str):
    return _serve_file(design_id, "thumbnail.png", "image/png")


@app.get("/api/designs/{design_id}/full.png")
def get_full(design_id: str):
    """The uncropped source image, for the interactive crop/pan/zoom UI."""
    return _serve_file(design_id, "candidate_full.png", "image/png")


@app.get("/api/designs/{design_id}/heightmap.png")
def get_heightmap(design_id: str):
    return _serve_file(design_id, "heightmap.png", "image/png", download_name="heightmap.png")


@app.get("/api/designs/{design_id}/preview.png")
def get_preview(design_id: str):
    return _serve_file(design_id, "preview.png", "image/png")


@app.get("/api/designs/{design_id}/model.stl")
def get_stl(design_id: str):
    from app.stl_export import heightmap_to_stl

    try:
        d = storage.design_dir(design_id)
    except ValueError:
        raise HTTPException(400, "Invalid design id")
    heightmap_path = d / "heightmap.png"
    if not heightmap_path.exists():
        raise HTTPException(404, "No processed heightmap yet for this design")

    stl_path = d / "model.stl"
    if not stl_path.exists() or stl_path.stat().st_mtime < heightmap_path.stat().st_mtime:
        heightmap_16bit = cv2.imread(str(heightmap_path), cv2.IMREAD_UNCHANGED)
        h_px, w_px = heightmap_16bit.shape  # the face size it was made for (PX_PER_MM px per mm)
        mesh = heightmap_to_stl(heightmap_16bit, face_width_mm=w_px / config.PX_PER_MM, face_height_mm=h_px / config.PX_PER_MM)
        mesh.export(str(stl_path))

    return FileResponse(str(stl_path), media_type="model/stl", filename="ring_face_relief.stl")


@app.get("/api/designs/{design_id}/params.json")
def get_params(design_id: str):
    d = storage.design_dir(design_id)
    path = d / "params.json"
    if not path.exists():
        raise HTTPException(404, "Not found")
    return JSONResponse(storage.read_json(path))


def _serve_file(design_id: str, filename: str, media_type: str, download_name: Optional[str] = None):
    try:
        d = storage.design_dir(design_id)
    except ValueError:
        raise HTTPException(400, "Invalid design id")
    path = d / filename
    if not path.exists():
        raise HTTPException(404, "Not found")
    kwargs = {"media_type": media_type}
    if download_name:
        kwargs["filename"] = download_name
    return FileResponse(str(path), **kwargs)


# --- static frontend ---------------------------------------------------

app.mount("/", StaticFiles(directory=str(config.STATIC_DIR), html=True), name="static")
