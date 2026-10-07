"""
Prompt lab (/lab.html): tune the relief prompt on a fixed set of test images.

- Sources: test images by category (people, animals, scenery, paintings,
  logos, drawings, objects), each in the "tune" or "holdout" split, listed in
  sources/sources.csv with where it came from and its licence.
- Prompt versions: prompts/v1.json, v2.json, ... -- the generation settings
  (model, style, options, faithfulness) and every piece of prompt text
  (relief_prompts.default_templates()). A version that has been run is locked,
  so a run always matches the version it names; changes go in a new version.
- Runs: one prompt version over a set of sources, each with a fixed seed per
  source (the same across versions, so differences come from the prompt).
  Each result gets the model's image, its depth map and a shaded preview of
  the relief that depth makes, plus meta.json (the exact prompt, seed, cost).
- Reviews: a verdict, a 1-5 score, error tags and a comment per result
  (review.json), totalled per run in summary.json.

Everything is plain files under config.PROMPT_LAB_DIR (README.md there
describes the layout), so the data can be read and analysed without the app.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from app import config, enhance, relief, relief_prompts

router = APIRouter(prefix="/api/lab")

# category id (also the prompt's image type) -> label
CATEGORIES = {
    "portrait": "People",
    "animal": "Animals",
    "landscape": "Scenery",
    "painting": "Paintings",
    "logo": "Logos",
    "drawing": "Drawings / cartoons",
    "object": "Objects",
}
SPLITS = ("tune", "holdout")
HOLDOUT_EVERY = 4               # "auto" split: every 4th upload in a category is held out
TARGET_PER_CATEGORY = 25
VERDICTS = ("good", "usable", "bad")
SOURCE_FIELDS = ["id", "category", "split", "file", "origin", "licence", "tags", "notes", "added"]
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
UPLOAD_TYPES = {"image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/webp": ".webp"}
WEBP_QUALITY = 92

DEFAULT_TAGS = [
    {"group": "Accuracy", "tags": ["identity_changed", "proportions_wrong", "missing_element", "extra_element",
                                   "cropped", "composition_changed"]},
    {"group": "Relief", "tags": ["too_flat", "too_deep", "mushy_detail", "noisy_texture", "depth_inverted"]},
    {"group": "Material & light", "tags": ["colour_left", "baked_shadows", "wrong_material", "photo_not_sculpture"]},
    {"group": "Background & options", "tags": ["background_not_removed", "border_ignored", "option_ignored"]},
    {"group": "Text & logos", "tags": ["text_garbled", "lines_lost", "levels_unclear"]},
]

README = """# Prompt lab data

Written by the SilverSignal prompt lab (/lab.html, app/prompt_lab.py). Everything
here is plain files; this describes them for anyone (or any model) analysing it.

## sources/
- `sources.csv`: one row per test image: `id`, `category` (portrait, animal,
  landscape, painting, logo, drawing, object -- also the prompt's image type),
  `split` (`tune`: used to tune prompts; `holdout`: kept back to check a prompt
  version generalises), `file` (path from here), `origin`, `licence`, `tags`,
  `notes`, `added`.
- `<category>/<id>.<ext>`: the images, exactly as uploaded.

## prompts/
`v<N>.json`, one per prompt version:
- `settings`: `model` (fal.ai model id), `style`, `additions`, `fidelity`.
- `templates`: every piece of prompt text (see app/relief_prompts.py,
  build_relief_prompt for how they are put together):
  `convert` (main instruction, `{style}` = the style's `relief` words),
  `keep_all` / `keep_subject` (faithfulness, `{except_changes}`), `casting`,
  `rules`, `negative`, `image_types` (one sentence per category), `styles`,
  `additions`, and `describe` (only for non-instruction models).
- `notes`: what changed and why; `based_on`: the version it was copied from.
A version with runs is locked. To propose a change, write a new
`v<N+1>.json`: copy the latest, edit `templates`/`settings`, set `based_on`
and explain the change in `notes`.

## runs/<run_id>/
- `run.json`: the prompt version, its settings, the source ids, timings, cost.
- `summary.json`: totals per category: verdict counts, average score, error tag counts.
- `<source_id>/`: one result:
  - `generation.webp`: the model's image (the relief, re-rendered from the source);
  - `depth.png`: the depth map the app turns into the ring's heightmap (white = raised);
  - `relief.webp`: a shaded render of that depth, roughly how the metal will look;
  - `meta.json`: exact prompt and negative prompt, model, seed (fixed per
    source, the same in every run), cost, time taken;
  - `review.json`: `verdict` (good / usable / bad), `score` (1-5), `errors`
    (tags from ../../error_tags.json), `comment`;
  - `error.json`: only if generation failed.

## error_tags.json
The error tags reviews can use, by group.
"""

_lock = threading.RLock()                          # sources.csv, prompt files, run.json/summary.json writes
_active: dict[str, dict] = {}                      # run_id -> {"stop": bool, "running": set(), "thread": Thread}


# --- paths & small helpers ------------------------------------------------------------

def lab_dir() -> Path:
    d = config.PROMPT_LAB_DIR
    for sub in ("sources", "prompts", "runs"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    if not (d / "README.md").exists():
        (d / "README.md").write_text(README)
    if not (d / "error_tags.json").exists():
        _write_json(d / "error_tags.json", DEFAULT_TAGS)
    return d


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _safe_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name or "") or name.startswith("."):
        raise HTTPException(400, f"Invalid name '{name}'")
    return name


def seed_for(source_id: str) -> int:
    """A fixed seed per source, so every prompt version sees the same noise."""
    return int(hashlib.sha1(source_id.encode()).hexdigest()[:8], 16) % 2_147_483_647


def load_rgb(path: Path) -> np.ndarray:
    """An upload as RGB: phone photos turned upright (EXIF), transparency on white."""
    img = ImageOps.exif_transpose(Image.open(path))
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba)
    return np.array(img.convert("RGB"))


def _cap(rgb: np.ndarray, max_edge: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = max_edge / max(h, w)
    return rgb if s >= 1 else cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def _save_webp(path: Path, rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_WEBP_QUALITY, WEBP_QUALITY])


def shaded_relief(depth: np.ndarray) -> np.ndarray:
    """A grey render of `depth` (0..1, 1 = raised) lit from the top left, as a
    low relief roughly 4% of the image's width high -- a quick look at the
    form the heightmap will have."""
    h, w = depth.shape
    height = cv2.GaussianBlur(depth.astype(np.float32), (0, 0), 0.8) * 0.04 * max(h, w)
    gy, gx = np.gradient(height)
    n = np.dstack([-gx, -gy, np.ones_like(gx)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    light = np.array([-0.5, -0.6, 0.62], np.float32)
    light /= np.linalg.norm(light)
    shade = 0.25 + 0.75 * np.clip(n @ light, 0, 1)
    return (np.clip(shade, 0, 1) * 255).astype(np.uint8)


# --- sources ---------------------------------------------------------------------------

def read_sources() -> list[dict]:
    path = lab_dir() / "sources" / "sources.csv"
    if not path.exists():
        return []
    with _lock, open(path, newline="") as fh:
        return [{k: row.get(k) or "" for k in SOURCE_FIELDS} for row in csv.DictReader(fh)]


def _write_sources(rows: list[dict]) -> None:
    path = lab_dir() / "sources" / "sources.csv"
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=SOURCE_FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    tmp.replace(path)


def _source(source_id: str) -> dict:
    row = next((r for r in read_sources() if r["id"] == source_id), None)
    if row is None:
        raise HTTPException(404, f"Unknown source '{source_id}'")
    return row


@router.get("/options")
def options():
    """Everything the page's menus need."""
    lab_dir()
    opts = relief_prompts.options()
    choices = [{"id": m, "label": label, "price_usd": relief.fal_price_usd(m, 1.0)}
               for m, label in config.FAL_MODEL_CHOICES] if config.RELIEF_PROVIDER == "fal" else []
    return {
        "categories": [{"id": k, "label": v} for k, v in CATEGORIES.items()],
        "target_per_category": TARGET_PER_CATEGORY,
        "splits": list(SPLITS), "verdicts": list(VERDICTS),
        "provider": config.RELIEF_PROVIDER, "models": choices,
        "default_model": config.FAL_MODEL if config.RELIEF_PROVIDER == "fal" else None,
        "styles": opts["styles"], "additions": opts["additions"],
        "tags": _read_json(lab_dir() / "error_tags.json", DEFAULT_TAGS),
        "lab_dir": str(config.PROMPT_LAB_DIR),
    }


@router.get("/sources")
def list_sources():
    return {"sources": read_sources()}


@router.post("/sources")
async def upload_sources(files: list[UploadFile] = File(...), category: str = Form(...), split: str = Form("auto"),
                         origin: str = Form(""), licence: str = Form(""), tags: str = Form("")):
    """Add test images to a category. split "auto" holds back every 4th one."""
    if category not in CATEGORIES:
        raise HTTPException(400, f"Unknown category '{category}'")
    if split not in (*SPLITS, "auto"):
        raise HTTPException(400, f"Unknown split '{split}'")
    added, errors = [], []
    for f in files:
        data = await f.read()
        ext = UPLOAD_TYPES.get(f.content_type or "")
        if ext is None:
            errors.append(f"{f.filename}: not a PNG, JPEG or WebP image")
            continue
        if len(data) > MAX_UPLOAD_BYTES:
            errors.append(f"{f.filename}: larger than {MAX_UPLOAD_BYTES // 1024 // 1024} MB")
            continue
        try:
            Image.open(io.BytesIO(data)).verify()
        except Exception:
            errors.append(f"{f.filename}: couldn't be read as an image")
            continue
        with _lock:
            rows = read_sources()
            in_cat = [r for r in rows if r["category"] == category]
            n = max([int(r["id"].rsplit("-", 1)[1]) for r in in_cat if r["id"].rsplit("-", 1)[1].isdigit()] or [0]) + 1
            sid = f"{category}-{n:03d}"
            the_split = split if split != "auto" else ("holdout" if len(in_cat) % HOLDOUT_EVERY == HOLDOUT_EVERY - 1
                                                       else "tune")
            rel = Path(category) / f"{sid}{ext}"
            dest = lab_dir() / "sources" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            row = {"id": sid, "category": category, "split": the_split, "file": rel.as_posix(),
                   "origin": origin.strip() or (f.filename or ""), "licence": licence.strip(), "tags": tags.strip(),
                   "notes": "", "added": datetime.now().isoformat(timespec="seconds")}
            rows.append(row)
            _write_sources(rows)
        added.append(row)
    return {"added": added, "errors": errors}


class SourcePatch(BaseModel):
    split: str | None = None
    origin: str | None = None
    licence: str | None = None
    tags: str | None = None
    notes: str | None = None


@router.patch("/sources/{source_id}")
def update_source(source_id: str, req: SourcePatch):
    if req.split is not None and req.split not in SPLITS:
        raise HTTPException(400, f"Unknown split '{req.split}'")
    with _lock:
        rows = read_sources()
        row = next((r for r in rows if r["id"] == source_id), None)
        if row is None:
            raise HTTPException(404, f"Unknown source '{source_id}'")
        for k, v in req.model_dump(exclude_none=True).items():
            row[k] = v.strip()
        _write_sources(rows)
    return row


@router.delete("/sources/{source_id}")
def delete_source(source_id: str):
    """Remove a test image (results already made from it stay in their runs)."""
    with _lock:
        rows = read_sources()
        row = next((r for r in rows if r["id"] == source_id), None)
        if row is None:
            raise HTTPException(404, f"Unknown source '{source_id}'")
        (lab_dir() / "sources" / row["file"]).unlink(missing_ok=True)
        _write_sources([r for r in rows if r["id"] != source_id])
    return {"deleted": source_id}


# --- prompt versions ------------------------------------------------------------------

def _version_num(v: str) -> int:
    m = re.fullmatch(r"v(\d+)", v)
    if not m:
        raise HTTPException(400, f"Invalid version '{v}'")
    return int(m.group(1))


def _versions() -> list[str]:
    files = (lab_dir() / "prompts").glob("v*.json")
    return sorted((p.stem for p in files if re.fullmatch(r"v\d+", p.stem)), key=_version_num)


def _runs_using(version: str) -> list[str]:
    return [p.parent.name for p in (lab_dir() / "runs").glob("*/run.json")
            if (_read_json(p, {}) or {}).get("version") == version]


def _default_version() -> dict:
    return {
        "version": "v1", "based_on": None, "created": datetime.now().isoformat(timespec="seconds"),
        "notes": "The app's built-in prompt (app/relief_prompts.py) at the time the lab was set up.",
        "settings": {"model": config.FAL_MODEL if config.RELIEF_PROVIDER == "fal" else None,
                     "style": relief_prompts.DEFAULT_STYLE, "additions": [],
                     "fidelity": config.RELIEF_FIDELITY_DEFAULT},
        "templates": relief_prompts.default_templates(),
    }


def read_version(version: str) -> dict:
    with _lock:
        if version == "v1" and not _versions():
            _write_json(lab_dir() / "prompts" / "v1.json", _default_version())
        data = _read_json(lab_dir() / "prompts" / f"{_safe_name(version)}.json")
    if data is None:
        raise HTTPException(404, f"Unknown prompt version '{version}'")
    return data


def _check_version(data: dict) -> None:
    """Every category's prompt must build (catches a broken {placeholder})."""
    s = data.get("settings") or {}
    if s.get("style") not in relief_prompts.STYLES:
        raise HTTPException(400, f"Unknown style '{s.get('style')}'")
    unknown = [a for a in s.get("additions") or [] if a not in relief_prompts.ADDITIONS]
    if unknown:
        raise HTTPException(400, f"Unknown additions: {', '.join(unknown)}")
    if not 0 <= float(s.get("fidelity", 0.8)) <= 1:
        raise HTTPException(400, "fidelity must be between 0 and 1")
    if s.get("model") and config.RELIEF_PROVIDER == "fal" and s["model"] not in dict(config.FAL_MODEL_CHOICES):
        raise HTTPException(400, f"Unknown model '{s['model']}' (add it to FAL_MODEL_CHOICES in app/config.py)")
    for cat in CATEGORIES:
        for instruction in (True, False):
            try:
                relief_prompts.build_relief_prompt(s["style"], s.get("additions") or [], "", instruction=instruction,
                                                   fidelity=float(s.get("fidelity", 0.8)), image_type=cat,
                                                   templates=data.get("templates"))
            except (KeyError, IndexError, ValueError, TypeError, AttributeError) as err:
                raise HTTPException(400, f"The prompt text doesn't build ({type(err).__name__}: {err}). "
                                         "Only {style} and {except_changes} placeholders are allowed.")


def build_prompt(version: dict, category: str, instruction: bool = True) -> tuple[str, str]:
    s = version["settings"]
    return relief_prompts.build_relief_prompt(s["style"], s.get("additions") or [], "", instruction=instruction,
                                              fidelity=float(s.get("fidelity", 0.8)), image_type=category,
                                              templates=version.get("templates"))


@router.get("/prompts")
def list_prompts():
    read_version("v1")  # makes v1 on first use
    out = []
    for v in _versions():
        data = read_version(v)
        runs = _runs_using(v)
        out.append({"version": v, "based_on": data.get("based_on"), "created": data.get("created"),
                    "notes": data.get("notes", ""), "settings": data.get("settings"), "runs": runs,
                    "locked": bool(runs)})
    return {"versions": out}


@router.get("/prompts/{version}")
def get_prompt(version: str):
    data = read_version(version)
    runs = _runs_using(version)
    try:
        provider_instruction = relief.get_relief_provider().instruction
    except relief.ReliefError:
        provider_instruction = True
    examples = {cat: build_prompt(data, cat, provider_instruction)[0] for cat in CATEGORIES}
    return {**data, "runs": runs, "locked": bool(runs), "examples": examples}


class NewVersion(BaseModel):
    based_on: str
    notes: str = ""


@router.post("/prompts")
def new_prompt(req: NewVersion):
    """A new version, copied from `based_on`."""
    base = read_version(req.based_on)
    with _lock:
        n = max([_version_num(v) for v in _versions()] or [0]) + 1
        data = {**base, "version": f"v{n}", "based_on": req.based_on,
                "created": datetime.now().isoformat(timespec="seconds"), "notes": req.notes}
        _write_json(lab_dir() / "prompts" / f"v{n}.json", data)
    return get_prompt(f"v{n}")


class VersionUpdate(BaseModel):
    notes: str = ""
    settings: dict
    templates: dict


@router.put("/prompts/{version}")
def save_prompt(version: str, req: VersionUpdate):
    data = read_version(version)
    if _runs_using(version):
        raise HTTPException(409, f"{version} has been run, so it's locked. Make a new version from it instead.")
    data.update(notes=req.notes, settings=req.settings, templates=req.templates)
    _check_version(data)
    with _lock:
        _write_json(lab_dir() / "prompts" / f"{_safe_name(version)}.json", data)
    return get_prompt(version)


# --- runs ----------------------------------------------------------------------------

def _run_dir(run_id: str) -> Path:
    d = lab_dir() / "runs" / _safe_name(run_id)
    if not (d / "run.json").exists():
        raise HTTPException(404, f"Unknown run '{run_id}'")
    return d


def _item_status(d: Path, run_id: str, sid: str) -> str:
    if (d / "generation.webp").exists():
        return "done"
    if sid in _active.get(run_id, {}).get("running", set()):
        return "running"
    if (d / "error.json").exists():
        return "failed"
    return "queued" if run_id in _active else "pending"


def _item_out(run_id: str, d: Path, sid: str, sources: dict) -> dict:
    meta = _read_json(d / "meta.json", {}) or {}
    src = sources.get(sid, {})
    base = f"runs/{run_id}/{sid}"
    return {
        "source_id": sid, "category": meta.get("category") or src.get("category"), "split": src.get("split"),
        "status": _item_status(d, run_id, sid),
        "error": (_read_json(d / "error.json", {}) or {}).get("error"),
        "source": f"sources/{src['file']}" if src else meta.get("source_file"),
        "generation": f"{base}/generation.webp", "depth": f"{base}/depth.png", "relief": f"{base}/relief.webp",
        "usd": meta.get("usd"), "seconds": meta.get("seconds"), "seed": meta.get("seed"),
        "prompt": meta.get("prompt"),
        "review": _read_json(d / "review.json"),
    }


class NewRun(BaseModel):
    version: str
    split: str = Field(default="tune", pattern="^(tune|holdout|all)$")
    categories: list[str] = []                     # none = every category
    source_ids: list[str] = []                     # given: exactly these (split/categories ignored)
    notes: str = ""


@router.post("/runs")
def start_run(req: NewRun):
    """Generate every chosen source with one prompt version, in the background."""
    version = read_version(req.version)
    _check_version(version)
    sources = read_sources()
    if req.source_ids:
        known = {r["id"] for r in sources}
        missing = [s for s in req.source_ids if s not in known]
        if missing:
            raise HTTPException(400, f"Unknown sources: {', '.join(missing)}")
        ids = list(dict.fromkeys(req.source_ids))
    else:
        ids = [r["id"] for r in sources
               if (req.split == "all" or r["split"] == req.split)
               and (not req.categories or r["category"] in req.categories)]
    if not ids:
        raise HTTPException(400, "No test images match that choice.")
    with _lock:
        base = f"{date.today().isoformat()}_{req.version}"
        run_id, k = base, 2
        while (lab_dir() / "runs" / run_id).exists():
            run_id, k = f"{base}_{k}", k + 1
        d = lab_dir() / "runs" / run_id
        d.mkdir(parents=True)
        _write_json(d / "run.json", {
            "run_id": run_id, "version": req.version, "settings": version["settings"],
            "provider": config.RELIEF_PROVIDER, "split": req.split, "categories": req.categories,
            "notes": req.notes, "created": datetime.now().isoformat(timespec="seconds"), "sources": ids,
        })
    _launch(run_id)
    return get_run(run_id)


def _launch(run_id: str, only: list[str] | None = None) -> None:
    """Start (or resume) a run's worker: makes every result not yet made."""
    with _lock:
        if run_id in _active:
            return
        state = {"stop": False, "running": set()}
        _active[run_id] = state
    t = threading.Thread(target=_work, args=(run_id, state, only), daemon=True, name=f"lab-{run_id}")
    state["thread"] = t
    t.start()


def _work(run_id: str, state: dict, only: list[str] | None) -> None:
    try:
        d = lab_dir() / "runs" / run_id
        run = _read_json(d / "run.json", {})
        version = read_version(run["version"])
        todo = [sid for sid in (only or run["sources"]) if not (d / sid / "generation.webp").exists()]
        try:
            provider = relief.get_relief_provider()
        except relief.ReliefError as err:
            for sid in todo:
                (d / sid).mkdir(exist_ok=True)
                _write_json(d / sid / "error.json", {"error": str(err), "at": time.time()})
            return
        sources = {r["id"]: r for r in read_sources()}
        with ThreadPoolExecutor(max_workers=max(1, config.PROMPT_LAB_PARALLEL)) as pool:
            list(pool.map(lambda sid: _generate_item(run_id, run, version, provider, sources.get(sid), sid, state),
                          todo))
    finally:
        with _lock:
            _active.pop(run_id, None)
        _write_summary(run_id)


def _generate_item(run_id: str, run: dict, version: dict, provider, src: dict | None, sid: str, state: dict) -> None:
    if state["stop"]:
        return
    d = lab_dir() / "runs" / run_id / sid
    d.mkdir(exist_ok=True)
    state["running"].add(sid)
    t0 = time.time()
    try:
        if src is None:
            raise relief.ReliefError("The source image has been deleted.")
        rgb = _cap(load_rgb(lab_dir() / "sources" / src["file"]), config.MAX_FULL_IMAGE_DIM_PX)
        prompt, negative = build_prompt(version, src["category"], provider.instruction)
        s = version["settings"]
        seed = seed_for(sid)
        img = provider.generate(rgb, prompt, negative, float(s.get("fidelity", 0.8)), seed, model=s.get("model"))
        relief.log_usage(provider.name, img, f"lab:{run_id}/{sid}", run_id)
        gen = _cap(img.rgb, config.MAX_FULL_IMAGE_DIM_PX)
        depth_error = None
        try:
            depth = enhance.estimate_depth(gen)
            cv2.imwrite(str(d / "depth.png"), np.round(depth * 255).astype(np.uint8))
            cv2.imwrite(str(d / "relief.webp"), shaded_relief(depth), [cv2.IMWRITE_WEBP_QUALITY, WEBP_QUALITY])
        except enhance.ModelUnavailable as err:
            depth_error = str(err)
        _write_json(d / "meta.json", {
            "run_id": run_id, "version": run["version"], "source_id": sid, "category": src["category"],
            "source_file": f"sources/{src['file']}", "provider": provider.name, "model": img.model,
            "settings": s, "prompt": prompt, "negative_prompt": negative, "seed": img.seed,
            "usd": img.usd, "seconds": round(time.time() - t0, 1), "depth_error": depth_error,
            "created": datetime.now().isoformat(timespec="seconds"),
        })
        _save_webp(d / "generation.webp", gen)   # last: its presence marks the result as done
        (d / "error.json").unlink(missing_ok=True)
    except Exception as err:  # one bad image mustn't stop the run
        _write_json(d / "error.json", {"error": str(err) or type(err).__name__, "at": time.time()})
    finally:
        state["running"].discard(sid)


@router.get("/runs")
def list_runs():
    out = []
    for p in sorted((lab_dir() / "runs").glob("*/run.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        run = _read_json(p, {}) or {}
        run_id = p.parent.name
        summary = _read_json(p.parent / "summary.json") or _summarise(run_id)
        out.append({"run_id": run_id, "version": run.get("version"), "created": run.get("created"),
                    "split": run.get("split"), "notes": run.get("notes", ""), "active": run_id in _active,
                    **{k: summary.get(k) for k in ("total", "done", "failed", "reviewed", "verdicts", "usd",
                                                    "avg_score")}})
    return {"runs": out}


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    d = _run_dir(run_id)
    run = _read_json(d / "run.json", {})
    sources = {r["id"]: r for r in read_sources()}
    items = [_item_out(run_id, d / sid, sid, sources) for sid in run["sources"]]
    return {**run, "active": run_id in _active, "items": items, "summary": _summarise(run_id)}


class RetryRequest(BaseModel):
    source_ids: list[str] = []      # none = every result not yet made


@router.post("/runs/{run_id}/resume")
def resume_run(run_id: str, req: RetryRequest):
    """Make the results that failed or never ran (e.g. after a restart)."""
    _run_dir(run_id)
    _launch(run_id, req.source_ids or None)
    return get_run(run_id)


@router.post("/runs/{run_id}/stop")
def stop_run(run_id: str):
    """Stop after the images being made now (resume later)."""
    _run_dir(run_id)
    if run_id in _active:
        _active[run_id]["stop"] = True
    return {"stopping": run_id in _active}


# --- reviews ---------------------------------------------------------------------------

class Review(BaseModel):
    verdict: str | None = Field(default=None, pattern="^(good|usable|bad)$")
    score: int | None = Field(default=None, ge=1, le=5)
    errors: list[str] = []
    comment: str = Field(default="", max_length=4000)


@router.put("/runs/{run_id}/items/{source_id}/review")
def save_review(run_id: str, source_id: str, req: Review):
    d = _run_dir(run_id)
    run = _read_json(d / "run.json", {})
    if source_id not in run.get("sources", []):
        raise HTTPException(404, f"'{source_id}' isn't in run {run_id}")
    item = d / _safe_name(source_id)
    if not (item / "generation.webp").exists():
        raise HTTPException(409, "This result hasn't been made yet.")
    review = {**req.model_dump(), "errors": list(dict.fromkeys(req.errors)),
              "reviewed_at": datetime.now().isoformat(timespec="seconds")}
    if review["verdict"] is None and review["score"] is None and not review["errors"] and not review["comment"].strip():
        (item / "review.json").unlink(missing_ok=True)
        review = None
    else:
        _write_json(item / "review.json", review)
    _write_summary(run_id)
    return {"review": review}


def _summarise(run_id: str) -> dict:
    d = lab_dir() / "runs" / run_id
    run = _read_json(d / "run.json", {}) or {}
    sources = {r["id"]: r for r in read_sources()}
    total = {"total": 0, "done": 0, "failed": 0, "reviewed": 0, "usd": 0.0,
             "verdicts": {v: 0 for v in VERDICTS}, "avg_score": None, "errors": {}, "by_category": {}}
    scores: dict[str, list[int]] = {}
    for sid in run.get("sources", []):
        item = d / sid
        meta = _read_json(item / "meta.json", {}) or {}
        cat = meta.get("category") or sources.get(sid, {}).get("category") or "unknown"
        c = total["by_category"].setdefault(cat, {"total": 0, "done": 0, "failed": 0, "reviewed": 0,
                                                  "verdicts": {v: 0 for v in VERDICTS}, "avg_score": None,
                                                  "errors": {}})
        for bucket in (total, c):
            bucket["total"] += 1
        if (item / "generation.webp").exists():
            total["done"] += 1
            c["done"] += 1
            total["usd"] += meta.get("usd") or 0
        elif (item / "error.json").exists():
            total["failed"] += 1
            c["failed"] += 1
        rv = _read_json(item / "review.json")
        if rv:
            for key, bucket in (("_all", total), (cat, c)):
                bucket["reviewed"] += 1
                if rv.get("verdict") in VERDICTS:
                    bucket["verdicts"][rv["verdict"]] += 1
                for tag in rv.get("errors") or []:
                    bucket["errors"][tag] = bucket["errors"].get(tag, 0) + 1
                if rv.get("score"):
                    scores.setdefault(key, []).append(rv["score"])
    total["usd"] = round(total["usd"], 4)
    for key, bucket in (("_all", total), *((k, v) for k, v in total["by_category"].items())):
        if scores.get(key):
            bucket["avg_score"] = round(sum(scores[key]) / len(scores[key]), 2)
        bucket["errors"] = dict(sorted(bucket["errors"].items(), key=lambda kv: -kv[1]))
    return {"run_id": run_id, "version": run.get("version"), **total}


def _write_summary(run_id: str) -> None:
    with _lock:
        _write_json(lab_dir() / "runs" / run_id / "summary.json", _summarise(run_id))


# --- error tags ------------------------------------------------------------------------

class NewTag(BaseModel):
    group: str = Field(..., min_length=1, max_length=60)
    label: str = Field(..., min_length=1, max_length=60)


@router.post("/tags")
def add_tag(req: NewTag):
    tag = re.sub(r"[^a-z0-9]+", "_", req.label.lower()).strip("_")
    if not tag:
        raise HTTPException(400, "The tag needs letters or numbers")
    with _lock:
        path = lab_dir() / "error_tags.json"
        groups = _read_json(path, DEFAULT_TAGS)
        if any(tag in g["tags"] for g in groups):
            return {"tags": groups, "tag": tag}
        group = next((g for g in groups if g["group"].lower() == req.group.strip().lower()), None)
        if group is None:
            group = {"group": req.group.strip(), "tags": []}
            groups.append(group)
        group["tags"].append(tag)
        _write_json(path, groups)
    return {"tags": groups, "tag": tag}


# --- files ---------------------------------------------------------------------------

@router.get("/files/{path:path}")
def get_file(path: str, w: int | None = None):
    """A file from the lab folder; with ?w=, a cached WebP at most w pixels wide/high."""
    root = lab_dir().resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root) or not target.is_file() or ".cache" in target.relative_to(root).parts:
        raise HTTPException(404, "Not found")
    if not w:
        return FileResponse(str(target))
    w = int(np.clip(w, 64, 2048))
    thumb = root / ".cache" / f"w{w}" / (target.relative_to(root).as_posix() + ".webp")
    if not thumb.exists() or thumb.stat().st_mtime < target.stat().st_mtime:
        thumb.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix == ".png" and "runs" in target.relative_to(root).parts:
            img = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)        # depth maps: keep them grey
            img = np.dstack([img] * 3) if img is not None else None
        else:
            img = load_rgb(target)
        if img is None:
            raise HTTPException(404, "Not found")
        _save_webp(thumb, _cap(np.ascontiguousarray(img), w))
    return FileResponse(str(thumb), media_type="image/webp")
