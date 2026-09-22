# Ring Face Relief Designer (MVP)

Type a natural-language description of a design, get a manufacturable raised-relief
heightmap for a fixed, flat signet ring face, and preview it in 3D as polished metal.

An AI image model (OpenAI's `gpt-image-2.5-flare`, or a built-in offline mock) only ever
produces a **2D image**. All geometry, manufacturing rules (minimum feature size, edge
margins, quantization), and the weight/volume estimate are deterministic code in
`app/processing.py` — no text-to-3D model is used anywhere.

Each click of "Generate" makes exactly **one** image (one billed API call) and adds it to
the gallery — there's no hidden batching. Click it again to add another for comparison.

The 3D preview renders the face attached to a small signet ring band so its real-world
scale is obvious — the engraved area is genuinely small (14mm x 12mm by default), and the
minimum-feature/edge-margin rules exist specifically because fine detail doesn't survive
manufacturing at that size. Once you pick a candidate, drag/zoom a crop box over the full
source image to choose exactly what lands on the face — see "Crop, pan & zoom" below.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # defaults to IMAGE_PROVIDER=mock, no API key needed

uvicorn app.main:app --reload --port 8420
```

Open http://127.0.0.1:8420 — type a description, click **Generate image** (once per
candidate you want), pick one, and tweak the sliders. Everything works fully offline
with the mock provider.

Run the test suite:

```bash
pytest -q
```

## Using real AI generation (OpenAI GPT Image 2.5)

1. Get an API key at https://platform.openai.com/api-keys. Note: OpenAI's image models
   require your organization to complete identity verification before they will generate
   images (Settings → Organization → General on platform.openai.com) — if generation
   fails with a permissions/access error, this is almost always why.
2. In `.env`: set `IMAGE_PROVIDER=openai` and `OPENAI_API_KEY=...`
3. Restart the server.

`OPENAI_IMAGE_MODEL` defaults to `gpt-image-2.5-flare` (cheaper/faster of the two current
GPT Image 2.5 variants; `gpt-image-2.5-sunburst` is the premium/higher-fidelity option at
the same per-token price, but tends to spend more tokens for more detail). **Do not use
`gpt-image-1`** — OpenAI has deprecated it, shutting down Dec 1, 2026.

`OPENAI_IMAGE_QUALITY` defaults to `medium` (`low`/`medium`/`high`/`xhigh`/`max`/`auto` —
each step up costs more and is slower; `low` is the single biggest cost lever if you're
optimizing for price over polish). To add another provider, implement `ImageProvider` in
`app/providers/` (see `app/providers/base.py`) and wire it into `get_provider()`.

Note: OpenAI's image API has a fixed set of output sizes (1024x1024, 1536x1024,
1024x1536) and no seed parameter — `app/providers/openai_provider.py` picks whichever
supported size is closest to the face's aspect ratio, and the existing cover-fit-resize
step in `/api/generate` crops it to the exact aspect ratio afterwards. The `seed` field
in `/api/generate` is accepted but has no effect with this provider.

## Changing the ring face size / resolution

Everything is in `app/config.py` — e.g. to make the face 16mm x 10mm at 60px/mm:

```python
FACE_WIDTH_MM = 16.0
FACE_HEIGHT_MM = 10.0
PX_PER_MM = 60
```

The heightmap resolution, edge-margin masking, and 3D viewer all derive from these
constants automatically. No other code changes needed.

## How it works

1. **Generate** (`POST /api/generate`): builds a locked prompt from a style preset +
   your text, asks the image provider for exactly one image, cover-fits it to the face
   aspect ratio, and caches it on disk as a candidate. Call it again (e.g. clicking
   "Generate" again in the UI) to add another candidate to the gallery — generation is
   never batched, so cost per click is predictable.
2. **Process** (`POST /api/process`, pure functions in `app/processing.py`): grayscale →
   optional invert/gamma/contrast → Gaussian blur → normalize → quantize to N levels →
   enforce minimum feature size (morphological open/close + small-component removal) →
   feathered edge margin → scale to the chosen relief height → 16-bit heightmap + 8-bit
   preview + a manufacturability report (coverage %, relief volume, estimated silver
   weight, warnings).
3. Moving a slider re-runs step 2 against the **cached** candidate image — the image
   model is never called again after the initial generation.
4. The frontend (`static/main.js`, Three.js, no build step) samples the 8-bit preview
   into a ~400x340 vertex grid, displaces a plane, and recomputes normals — polished
   with `MeshStandardMaterial` (metalness 1, roughness 0.08) under `RoomEnvironment`
   lighting. The exaggeration slider only scales the *displayed* mesh; exports always
   use the true (max 0.4mm) relief height. The face plate sits on a stylized torus
   band (see "The 3D ring preview" below) so its small scale reads clearly.

### Crop, pan & zoom

A candidate's **full, uncropped** source image is kept on disk (`candidate_full.png`,
capped to `MAX_FULL_IMAGE_DIM_PX` = 1600px on the long edge) — the face only ever shows
a crop of it, chosen interactively:

- **Zoom** (1x–6x) shrinks the cropped region, so more of the image maps onto the small
  face at higher effective detail (useful once you've composed a design and want to
  frame just the interesting part tightly).
- **Drag the crop box** directly on the source image to pick what's centered on the face
  (pan), independent of zoom.

Both are sent as `crop_zoom`/`crop_offset_x`/`crop_offset_y` to `/api/process`, applied
fresh against the cached full image every time (`app/imaging.py::cover_fit_resize`) —
adjusting them never re-calls the image model, and the on-screen crop box is computed
with the exact same math as the backend (`computeCropGeometry` in `main.js` mirrors
`cover_fit_resize` in Python) so what you see is what gets cropped.

### The 3D ring preview

The viewer attaches the face to a torus band (`RING_DIAMETER_MM` = 18mm, a plausible
finger size) purely so the face's real scale is obvious next to something recognizably
ring-sized — **this is not a manufacturing model of the shank.** The exported heightmap
and STL only ever describe the flat face; the manufacturer determines the actual
band/shank/finger-size geometry separately and fuses or engraves the face pattern onto it.

### Style presets

| Preset | Look | Default levels | Default blur |
|---|---|---|---|
| `emblem` | bold flat silhouette, tiered relief | 4 | 0 |
| `lineart` | bold outline engraving | 2 | 0 |
| `organic` | continuous sculpted bas-relief | 0 (continuous) | 0.15mm |

### Prompting the image model for good heightmaps

The locked templates (`app/prompts.py`) are written specifically so the *output image*
survives the processing pipeline well, not just so it looks good on its own:

- **Pure, perfectly flat black background.** A photographic background or soft vignette
  reads as low-level "raised" texture across the whole face once converted to grayscale.
- **Explicitly non-photographic rendering** ("flat vector illustration" / "matte clay
  medallion", "no photographic texture", "no noise", "no film grain"). Photo textures
  (skin, fabric weave, grain) all become unwanted high-frequency relief noise.
- **Thick, simple, well-separated shapes** well above `MIN_FEATURE_MM` (0.25mm), so the
  minimum-feature morphology step doesn't eat fine detail — a common failure mode is a
  intricate/thin design losing most of its detail to that step (the report warns if this
  happens).
- **A single centered subject**, since designs aren't auto-recentered.
- **No text/letters/watermark/border/frame**, since (a) diffusion models render text
  poorly at this scale and (b) our processing never attempts OCR/text-cleanup.

Note: OpenAI's Images API has no `negative_prompt` input, so all "avoid X" guidance above
is baked directly into the positive prompt string rather than a separate negative-prompt
field.

If your description mentions text/lettering (checked with a simple keyword list — "text",
"letters", "name", "monogram", etc.), the report includes a warning that this MVP doesn't
support legible text on the relief. This is a constraint of the *processing pipeline*, not
the image model: OpenAI's image models are actually quite good at rendering legible text, but any
text it draws gets flattened/quantized/min-feature-filtered along with everything else and
reliably comes out as illegible blobs once converted to relief — so the prompt templates
explicitly ask for none, and no attempt is made to strip or repair it if it appears anyway.

### Uploading your own photo/artwork

`POST /api/upload` (multipart file, PNG/JPEG/WEBP, 15MB max) skips the AI step entirely:
the image is cover-fit to the face aspect ratio and stored exactly like a generated
candidate, so it flows through the identical `/api/process` pipeline, sliders, and
report. Useful for scanned line art, logos, or photos with clear silhouettes.

## API

- `GET /api/config` — face/relief constants and presets, so the frontend never hardcodes them.
- `POST /api/generate {prompt, preset, seed?}` → one candidate id + thumbnail URL (always
  exactly one image per call — call it again to add another candidate).
- `POST /api/upload` (multipart `file`) → one candidate id + thumbnail URL, from a user photo.
- `POST /api/process {candidate_id, invert?, gamma?, contrast?, blur_mm?, levels?, min_feature_mm?, relief_height_mm?, crop_zoom?, crop_offset_x?, crop_offset_y?}`
  → preview/heightmap/params URLs + manufacturability report. Never calls the image model;
  crop/zoom/pan is re-applied to the cached full image on every call.
- `GET /api/designs/{id}/heightmap.png` — 16-bit grayscale PNG (0..65535 = 0..RELIEF_MAX_MM).
- `GET /api/designs/{id}/preview.png` — 8-bit grayscale, used by the 3D viewer.
- `GET /api/designs/{id}/full.png` — the uncropped source image, for the crop/pan/zoom UI.
- `GET /api/designs/{id}/params.json` — prompt, preset, seed, model, all processing params
  (crop params included), report.
- `GET /api/designs/{id}/model.stl` — **(stretch)** watertight STL: base slab
  (`BASE_THICKNESS_MM` = 1.0mm) + relief, built from the 16-bit heightmap and verified
  watertight with `trimesh` in `tests/test_stl_export.py`. Grid is downsampled to a
  ~220px max dimension to keep face count/file size reasonable — still far finer than
  `MIN_FEATURE_MM`.

## Storage

Filesystem only, `./data/designs/<uuid>/` — no database, no auth. Each candidate
(generated or uploaded) gets its own folder: `candidate_full.png` (the full, uncropped
source image), `thumbnail.png`, `meta.json` (prompt/preset/seed/provider), and after
processing, `heightmap.png`, `preview.png`, `params.json`, and (on request) `model.stl`.

## Assumptions made

- A generated **candidate's id doubles as the eventual design id** — `/api/generate`
  creates one design folder per call (each call = exactly one image = one billed API
  request, by design, to keep cost predictable), rather than a batch id you then index
  into. This is what makes "re-processing never re-calls the image model" simple: the
  candidate image is just sitting in that folder already.
- The edge-margin **feather is intentionally continuous** even when `levels >= 2`
  (tiered/quantized mode) — the spec asks for a 0.3mm feather, which is inherently a
  ramp, not a hard cutoff. Only the *interior* (inset past margin+feather) is guaranteed
  to have exactly N discrete levels; this is covered by `tests/test_processing.py`.
- Generated images are requested at the face aspect ratio, snapped to whichever of
  OpenAI's fixed output sizes (1024x1024 / 1536x1024 / 1024x1536) is closest, and kept
  in full (capped to `MAX_FULL_IMAGE_DIM_PX`) rather than immediately cropped — the
  face-aspect-ratio crop happens per `/api/process` call instead, using the crop_*
  params, so pan/zoom can be adjusted after the fact without a re-generation call.
- The crop box's default (`zoom=1, offset=(0,0)`) reproduces the old fixed
  cover-fit-and-crop behavior exactly, so an unadjusted candidate processes identically
  to before this feature existed.
- The 3D ring band (torus, `RING_DIAMETER_MM`/`RING_BAND_THICKNESS_MM`) is a
  proportional visual aid only — picked to make the face's small scale legible, not
  derived from any real ring-sizing standard or sent to a manufacturer.
- No path traversal protection was needed beyond validating design ids are the uuid4 hex
  strings we generate ourselves (`storage._safe_id`), since there's no auth/multi-tenant
  concern in this MVP.
- The 3D viewer's bezel (rim width, slab thickness) is a fixed cosmetic choice (1.6mm)
  to suggest a ring band visually — it's independent of the STL export's
  `BASE_THICKNESS_MM` (1.0mm) constant, which is what actually gets exported.
- Weight/volume in the report account for the **relief only** (matches the spec), not the
  base plate — a real ring's total silver weight would also include the plain base slab
  and shank, which are out of scope for this MVP.

## Known limitations

- No text/lettering support (by design — diffusion models render text poorly, and this
  MVP has no OCR/text-repair step).
- `MockProvider` images are simple procedural shapes (rings, blobs, silhouettes),
  loosely seeded from a hash of the prompt — they don't reflect the prompt's actual
  content. That's expected; it exists purely so the app/tests work with zero setup.
- The STL export downsamples the heightmap grid for a manageable file size/face count;
  it's a preview-quality mesh, not a full-resolution manufacturing file.
- The 3D ring band is a stylized proportional preview (see "The 3D ring preview" above),
  not a CAD-accurate shank — only the face relief is ever exported.
