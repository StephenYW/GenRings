# Ring Face Relief Designer (MVP)

Type a natural-language description of a design, get a manufacturable raised-relief
heightmap for the top of a signet ring head, and preview it in 3D as polished metal.

An AI image model (OpenAI's `gpt-image-2.5-flare`, or a built-in offline mock) only ever
produces a **2D image**. All geometry, manufacturing rules (minimum feature size, edge
margins, quantization), and the weight/volume estimate are deterministic code in
`app/processing.py` — no text-to-3D model is used anywhere.

Each click of "Generate" makes exactly **one** image (one billed API call) and adds it to
the gallery — there's no hidden batching. Click it again to add another for comparison.

The 3D preview shows real signet ring models from a library of 10 shapes in UK sizes H–Z
(pick one from the menu), and applies the design to two of them: **S Square**, over its
flat top (about 13.1mm x 13.4mm at the default 8°, adjustable with a slider), and **S
Square (ridged)**, over the recessed floor inside its ridge (11.9mm x 12.0mm). Sizes come
from `static/rings/relief.json`.
The minimum-feature rules exist because fine detail doesn't survive manufacturing at that
size. Once you pick a candidate, drag/zoom a crop box over the full
source image to choose exactly what lands on the face — see "Crop, pan & zoom" below.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # defaults to IMAGE_PROVIDER=mock, no API key needed

python tools/prepare_rings.py   # once: converts the ring library (see "Ring library")

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

## Ring library

The ring models live in `3D Models/Rings STL/<Shape>/<Size>_<Shape>.stl` (not committed,
1.5GB): 10 shapes — Circle, Oval, Square, Rectangle and Thin rectangle, each plain or
ridged — in UK sizes H–Z (inner diameter 14.7mm to 22.0mm), all clean watertight meshes in
millimetres. `tools/prepare_rings.py` (run it with the project venv; about a minute)
converts them for the viewer:

- `static/rings/<Shape>/<Size>.glb` — one mesh per ring, rotated to the viewer's axes
  (up = +Y, finger hole along Z). Not committed (~570MB); rerun the script to rebuild.
- `static/rings/catalog.json` — every shape and size with its measured inner diameter,
  which the viewer's ring menu (Shape / Ridged / Size) is built from.
- `static/rings/relief.json` — the design rings' design areas, one entry per ring
  (`rings`, keyed "Shape/Size", each with a `mode`), plus the `default` ring.

Every ring gets the same polished silver (see "Metal look").

Each run of the script stamps a build `version` into `catalog.json` and `relief.json`,
and the viewer requests models as `<Size>.glb?v=<version>` (and the JSON uncached), so
a browser can't pair a cached model from an older build with newer design-area data.
If a design ring's model still doesn't match its data (vertex count), the viewer shows
an error rather than applying the design to the wrong vertices.

Until a design is picked, each design ring shows a placeholder fitted to its design area:
concentric copies of the area's outline, the outermost filling it to the edge (on the
ridged ring, right up to the ridge walls).

### Where the design goes

Designs go on the rings listed in `RELIEF_RINGS` in the script, each with its own way of
finding the design area. Each `/api/process` request says which ring the heightmap is
for (`relief_ring`), and the backend sizes it to that ring's area
(`config.face_geometry`). Switching between design rings in the viewer re-makes the
heightmap at the new size.

**S Square (ridged) — "recess".** The design fills the recessed floor inside the ridge:
the faces connected to the centre of the top that tilt less than `RECESS_MAX_TILT_DEG`
(60°), which takes in the flat floor and the small fillet where it meets the ridge's
vertical inner wall, so it uses all the space inside the ridge (11.94mm x 11.97mm). The
floor is raised half way up the ridge's inner wall (`RECESS_FLOOR_RAISE` = 0.5, 0.45mm of
the 0.9mm wall) for a shallower recess than the source model's, the wall shortening to
match. The floor and the ridge's inner wall are refined twice and come first in the GLB. The viewer
moves the floor straight up/down by the heightmap (top-down UVs over its box), so its
edge stays directly under the wall. Each wall vertex carries `_wall` = (the floor-edge
vertex below it, how far up the wall it is), and the viewer stretches the wall so its
foot follows that floor edge and its top stays at the ridge: when the design sinks the
floor (e.g. dark edges with the preview exaggeration up), the wall reaches down to meet
it instead of floating. There is no band transition outside the ridge.

**S Square — "tilt".** The design goes on its flat top: the triangles visible from straight above (a top-down
z-buffer) that tilt less than an angle you set live with the **Design area** slider under
the ring menu (0.5°–30°, default `DEFAULT_TILT_DEG` = 8°). Lower keeps the design on the
flattest part of the top; higher spreads it onto the rounded edge. The area's footprint
from above sets the heightmap's size and the crop box's aspect ratio (about 13.1mm x
13.4mm at 8°). The design is never clipped to that outline. The slider only shows for the
relief ring.

So the viewer can change the angle without a rebuild, the script prepares everything any
angle up to `TILT_MAX_DEG` (30°) could need:

- the "zone" — the top faces up to 30° plus the band around them (the rounded edge and
  upper shoulders, within `BLEND_MM` = 2.5mm, not facing down, so the finger hole is left
  alone) — refined twice (each pass splits every triangle in four, down to ~0.04mm edges),
  splitting the neighbouring triangles to match so there are no cracks. In the GLB the
  top faces come first, then the band's, and the zone's vertices come first;
- `relief.json`'s `tilt_table`: for every 0.5°, the area's bounding box seen from above
  and its outline. The backend sizes the heightmap from it (`face_tilt_deg` on
  `/api/process`, `config.face_geometry`), and the viewer maps its top-down UVs onto the
  same box (u along X, v along Z, image row 0 at -Z).

When the slider moves, the viewer (`computeDesignArea` in `static/main.js`, about 0.1s)
takes the top faces tilting less than the angle as the design area, traces its edge into
an ordered loop, and measures every nearby vertex's distance to that edge along the
surface (Dijkstra from all edge points at once). On release, it asks the backend for a
heightmap of the new size; while dragging, the current heightmap is stretched to the new
area.

So the design doesn't start abruptly at the area's edge, the band around it curves up to
meet it. The viewer (`displaceFace` in `static/main.js`) then, on every update:

1. reads the design's height at every point on the area's edge, smoothed only slightly
   along the edge (`EDGE_SMOOTH_MM` = 0.1mm, just to drop pixel noise), so the band
   follows the design's detail;
2. pushes each design-area vertex out along its normal (straight up) by the heightmap,
   easing into that smoothed edge height over the last `INNER_MM` (0.3mm), so the two
   meet exactly;
3. moves each band vertex along its own normal by the edge height at its nearest edge
   point times `coveProfile(t)`, with t = distance from the edge / `BLEND_MM`:
   `(1 - t)^p - cut * 6.75 * t * (1 - t)^2`. That is a concave cove: going up from the
   band, the surface dips into a hollow carved into the band (deepest, `COVE.cut` x the
   edge height, a third of the way down) and then sweeps up to meet the design's edge,
   more sharply the larger `p` — the **Edge cove steepness** slider (1–8, default 3).
   Everything is 0 with zero slope `BLEND_MM` down, so it blends into the untouched ring.

Because the cove scales with the design's height at the edge, there is no cove where
the design has no height there: the band stays exactly as it is. Other rings show without a design, and the menu says so.
The STL export is still the plain rectangular relief slab.

## How it works

1. **Generate** (`POST /api/generate`): builds a locked prompt from a style preset +
   your text, asks the image provider for exactly one image, cover-fits it to the face
   aspect ratio, and caches it on disk as a candidate. Call it again (e.g. clicking
   "Generate" again in the UI) to add another candidate to the gallery — generation is
   never batched, so cost per click is predictable.
2. **Process** (`POST /api/process`, pure functions in `app/processing.py`): grayscale →
   optional invert/gamma/contrast → Gaussian blur → normalize → quantize to N levels →
   enforce minimum feature size (morphological open/close + small-component removal) →
   scale to the chosen relief height → 16-bit heightmap + 8-bit
   preview + a manufacturability report (coverage %, relief volume, estimated silver
   weight, warnings).
3. Moving a slider re-runs step 2 against the **cached** candidate image — the image
   model is never called again after the initial generation.
4. The frontend (`static/main.js`, Three.js, no build step) applies the 8-bit preview
   to the relief ring (see "Where the design goes") and recomputes normals — polished
   with `MeshStandardMaterial` (metalness 1, roughness 0.05) in a photo-studio HDRI
   (see "Metal look" below). The exaggeration slider only changes the *displayed* mesh, and grows the
   design downwards: its highest point stays at its true height and everything below
   sinks `exaggeration` times deeper (height = top + (h − top) × exaggeration), so it
   deepens into the ring rather than rising out of it (on the ridged ring it stays below
   the ridge);
   exports always use the true (max 0.4mm) relief height.

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
- There is no edge margin by default (`EDGE_MARGIN_MM = 0`): the design runs right to the
  face's edge and is never clipped to its outline. If a margin is configured, its feather
  is intentionally continuous even when `levels >= 2` — only the interior is guaranteed
  to have exactly N discrete levels; this is covered by `tests/test_processing.py`.
- Generated images are requested at the face aspect ratio, snapped to whichever of
  OpenAI's fixed output sizes (1024x1024 / 1536x1024 / 1024x1536) is closest, and kept
  in full (capped to `MAX_FULL_IMAGE_DIM_PX`) rather than immediately cropped — the
  face-aspect-ratio crop happens per `/api/process` call instead, using the crop_*
  params, so pan/zoom can be adjusted after the fact without a re-generation call.
- The crop box's default (`zoom=1, offset=(0,0)`) reproduces the old fixed
  cover-fit-and-crop behavior exactly, so an unadjusted candidate processes identically
  to before this feature existed.
- No path traversal protection was needed beyond validating design ids are the uuid4 hex
  strings we generate ourselves (`storage._safe_id`), since there's no auth/multi-tenant
  concern in this MVP.
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
- Designs are applied to two rings (S Square and S Square ridged) so far; the other 188
  rings show without one.
- Only the face relief is ever exported (as a rectangular slab STL), not the ring it sits on.

## Metal look

The whole ring (body, face, relief) uses one shared sterling-silver material (`SILVER` in
`static/main.js`), buffed to a near-mirror polish (roughness 0.05). Raise `SILVER.roughness`
(0.15-0.3) for a satin finish.

The ring's surroundings are a real photo-studio HDRI (`loadStudio`, settings in `STUDIO`):
Poly Haven's "Monochrome Studio 02" (CC0), from `3D Models/monochrome_studio_02_4k.exr`
(not committed, 77MB; download the 4K EXR from https://polyhaven.com/a/monochrome_studio_02),
converted by `tools/prepare_skybox.py` (run it with the project venv after changing the
source) into `static/env/studio_env.hdr`. It's both what the silver reflects (its strip
softboxes and octabox give the highlights; its dark ceiling gives contrast) and the
background, slightly blurred (`backgroundBlur`) as if the camera were focused on the ring.
The script turns the panorama by `YAW_DEG` so the white seamless backdrop sits behind the
ring from the starting camera; three.js r160 can't rotate an environment map at runtime.
The HDRI loads in the background; until it arrives the viewer shows a plain light-gray
background. `STUDIO.reflectionGain` scales how strongly the metal reflects the studio.

The studio's own ceiling is dark, so an upward-facing face would mirror black. A hidden
soft lightbox above the ring (`STUDIO.topLight`: size, position, brightness) fixes that. It
sits on `REFLECTION_LAYER`, which the viewing camera never renders, so it only shows up in
the metal. It glows from its centre and fades smoothly to black at its edges
(`lightEdgeFade`), like a real softbox; a hard edge made tiny wobbles in the AI-generated
ring mesh show up as jagged reflection outlines. The reflections are captured once from the
ring's position (studio plus lightbox, ring hidden), so spinning the ring needs no recapture.

Two things keep the design readable on polished silver, which has no shading of its own:

- The top lightbox is graduated (`topLight.gradient`): bright at one end, dim at the other.
  Each slope of the relief mirrors a different brightness, so the design reads as shading
  rather than washing out to uniform white.
- The face's recesses are darkened like an oxidized ("antiqued") signet ring (`PATINA`):
  wherever the relief dips below its surroundings within `radiusMm`, the metal gets darker,
  up to `darkness`. Crevices beside raised detail go dark while broad flat areas stay
  polished. It's computed from the heightmap on every update and fades out near the rim so
  the face meets the body without a line. Set `darkness` to 0 for a plain polished face.

Viewer controls (`setupDragControls`): the camera starts turned slightly right
(`CAMERA_YAW`). Dragging left/right spins the ring on the spot like a turntable (the camera
and box stay put, so reflections sweep across the metal), dragging up/down tilts the camera
over or under the ring (all the way round to its underside), and the mouse wheel zooms.
