# Ring Face Relief Designer (MVP)

Type a natural-language description of a design, get a manufacturable raised-relief
heightmap for a fixed, flat signet ring face, and preview it in 3D as polished metal.

An AI image model (OpenAI's `gpt-image-1`, or a built-in offline mock) only ever produces
a **2D image**. All geometry, manufacturing rules (minimum feature size, edge margins,
quantization), and the weight/volume estimate are deterministic code in `app/processing.py`
— no text-to-3D model is used anywhere.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # defaults to IMAGE_PROVIDER=mock, no API key needed

uvicorn app.main:app --reload --port 8420
```

Open http://127.0.0.1:8420 — type a description, click **Generate candidates**, pick one,
and tweak the sliders. Everything works fully offline with the mock provider.

Run the test suite:

```bash
pytest -q
```

## Using real AI generation (OpenAI gpt-image-1)

1. Get an API key at https://platform.openai.com/api-keys. Note: `gpt-image-1` requires
   your OpenAI organization to complete identity verification before it will generate
   images (Settings → Organization → General on platform.openai.com) — if generation
   fails with a permissions/access error, this is almost always why.
2. In `.env`: set `IMAGE_PROVIDER=openai` and `OPENAI_API_KEY=...`
3. Restart the server.

`OPENAI_IMAGE_MODEL` defaults to `gpt-image-1`; `OPENAI_IMAGE_QUALITY` defaults to
`medium` (`low`/`medium`/`high`/`auto` — higher costs more and is slower, useful once
you've narrowed down a design and want a sharper final candidate). To add another
provider, implement `ImageProvider` in `app/providers/` (see `app/providers/base.py`)
and wire it into `get_provider()`.

Note: `gpt-image-1`'s API has a fixed set of output sizes (1024x1024, 1536x1024,
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
   your text, asks the image provider for `n_candidates` images, cover-fits each to the
   face aspect ratio, and caches them on disk as candidates.
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
   use the true (max 0.4mm) relief height.

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

Note: OpenAI's Images API (`gpt-image-1`) has no `negative_prompt` input, so all "avoid X"
guidance above is baked directly into the positive prompt string rather than a separate
negative-prompt field.

If your description mentions text/lettering (checked with a simple keyword list — "text",
"letters", "name", "monogram", etc.), the report includes a warning that this MVP doesn't
support legible text on the relief. This is a constraint of the *processing pipeline*, not
the image model: `gpt-image-1` is actually quite good at rendering legible text, but any
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
- `POST /api/generate {prompt, preset, n_candidates?, seed?}` → candidate ids + thumbnail URLs.
- `POST /api/upload` (multipart `file`) → one candidate id + thumbnail URL, from a user photo.
- `POST /api/process {candidate_id, invert?, gamma?, contrast?, blur_mm?, levels?, min_feature_mm?, relief_height_mm?}`
  → preview/heightmap/params URLs + manufacturability report. Never calls the image model.
- `GET /api/designs/{id}/heightmap.png` — 16-bit grayscale PNG (0..65535 = 0..RELIEF_MAX_MM).
- `GET /api/designs/{id}/preview.png` — 8-bit grayscale, used by the 3D viewer.
- `GET /api/designs/{id}/params.json` — prompt, preset, seed, model, all processing params, report.
- `GET /api/designs/{id}/model.stl` — **(stretch)** watertight STL: base slab
  (`BASE_THICKNESS_MM` = 1.0mm) + relief, built from the 16-bit heightmap and verified
  watertight with `trimesh` in `tests/test_stl_export.py`. Grid is downsampled to a
  ~220px max dimension to keep face count/file size reasonable — still far finer than
  `MIN_FEATURE_MM`.

## Storage

Filesystem only, `./data/designs/<uuid>/` — no database, no auth. Each candidate
(generated or uploaded) gets its own folder: `candidate.png` (raw), `thumbnail.png`,
`meta.json` (prompt/preset/seed/provider), and after processing, `heightmap.png`,
`preview.png`, `params.json`, and (on request) `model.stl`.

## Assumptions made

- A generated **candidate's id doubles as the eventual design id** — `/api/generate`
  returns 4 (by default) separate design folders up front, one per candidate, rather than
  a batch id you then index into. This is what makes "re-processing never re-calls the
  image model" simple: the candidate image is just sitting in that folder already.
- The edge-margin **feather is intentionally continuous** even when `levels >= 2`
  (tiered/quantized mode) — the spec asks for a 0.3mm feather, which is inherently a
  ramp, not a hard cutoff. Only the *interior* (inset past margin+feather) is guaranteed
  to have exactly N discrete levels; this is covered by `tests/test_processing.py`.
- Generated images are requested at the face aspect ratio, snapped to whichever of
  `gpt-image-1`'s fixed output sizes (1024x1024 / 1536x1024 / 1024x1536) is closest,
  then cover-fit resized/cropped to the exact heightmap pixel size — rather than
  requesting the exact (odd, e.g. 700x600) heightmap resolution directly, which the API
  doesn't support anyway.
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
