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
floor and the ridge's inner wall are refined twice and come first in the GLB. The viewer
moves the floor straight up/down by the heightmap (top-down UVs over its box), so its
edge stays directly under the wall, and lifts the design so its **highest point is
always level with the top of the ridge's inner wall** (`wall_top_y_mm` − `floor_y_mm`
in `relief.json`, 0.96mm): each point sits at wall top − (design top − h) x relief
scale x exaggeration. So at any exaggeration the peaks stay flush with the ridge and
only what's below them deepens. Each wall vertex carries `_wall` = (the floor-edge
vertex below it, how far up the wall it is), and the viewer stretches the wall so its
foot follows that floor edge and its top stays at the ridge: when the design sinks the
floor (e.g. dark edges with the preview exaggeration up), the wall reaches down to meet
it instead of floating. There is no band transition outside the ridge.

The **Ridge wall** slider (shown on this ring; 0–0.6mm, default 0.3mm) slides the ridge's
inner wall outward: a thinner rim and a bigger floor for the design (at 0.3mm the rim goes
from 1.22mm to 0.92mm thick and the floor from 11.94mm to 12.54mm across). The mesh is
never cut or remeshed: the prep script gives each point on top of the head near the floor's
edge a signed distance to that edge and an outward direction (`_ridge`, from a smoothed
signed-distance map of the floor's footprint), and the viewer (`ridgeShiftAt`) slides
points outward along it — rigidly for the wall and its fillets, fading smoothly to nothing
across the floor (which stretches) and the ridge's top and outer side (which compress), so
the ring's outside is unchanged. Points never pass one another, so no triangle flips
(checked up to 0.6mm; it starts to fold past ~0.7mm). The design area, heightmap
(`ridge_shift_mm` on `/api/process`) and crop box grow with the floor.

**S Square — "tilt".** The design goes on its flat top: the triangles visible from straight above (a top-down
z-buffer) that tilt less than an angle you set live with the **Design area** slider under
the ring menu (0.5°–30°, default `DEFAULT_TILT_DEG` = 8°). Lower keeps the design on the
flattest part of the top; higher spreads it onto the rounded edge. The area's footprint
from above sets the heightmap's size and the crop box's aspect ratio (about 13.1mm x
13.4mm at 8°). The design is never clipped to that outline. The slider only shows for the
relief ring.

So the viewer can change the angle without a rebuild, the script prepares everything any
angle up to `TILT_MAX_DEG` (30°) could need:

- the "zone" — the top faces up to 30° — refined twice (each pass splits every triangle
  in four, down to ~0.04mm edges), splitting the neighbouring triangles to match so there
  are no cracks. In the GLB the zone's faces and vertices come first;
- `relief.json`'s `tilt_table`: for every 0.5°, the area's bounding box seen from above
  and its outline. The backend sizes the heightmap from it (`face_tilt_deg` on
  `/api/process`, `config.face_geometry`), and the viewer maps its top-down UVs onto the
  same box (u along X, v along Z, image row 0 at -Z).

When the slider moves, the viewer (`computeDesignArea` in `static/main.js`) takes the top
faces tilting less than the angle as the design area. On release, it asks the backend for
a heightmap of the new size; while dragging, the current heightmap is stretched to the new
area.

The viewer (`displaceFace`) pushes each design-area vertex out along its normal (straight
up) by the heightmap. Nothing outside the area moves: the rounded edge and band stay
exactly as the source ring, and the design ends at the area's edge with a hard edge. Other rings show without a design, and the menu says so.
**Download ring STL** (top right of the 3D view, and in the report's downloads) saves the
whole ring as shown, with the design at its true height (never the preview exaggeration),
as a binary STL in millimetres, built in the browser from the displayed mesh (watertight;
its detail is the ring mesh's resolution). The face-slab STL below is still available.

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

### Removing the background

The **Remove background** checkbox keeps only the image's main subject, on either design
ring. A segmentation model (rembg, `BG_REMOVAL_MODEL` in `app/config.py`, default
`isnet-general-use`, ~180 MB, downloaded once to `~/.rembg` and run locally on the CPU in
about a second) finds the subject in the candidate's full image; the mask is cached as
`subject_mask.png` next to it, so crop/zoom/pan and slider changes never re-run it. On each
`/api/process` with `remove_background`, the mask gets the same crop and flips as the image,
then `apply_subject_mask` flattens the background to zero and re-stretches the subject's
own tones to `SUBJECT_BASE_LEVEL`..1 (0.2..1), so the subject stands on a raised plateau
with its full range of detail. (Invert, which runs first, flips the subject's own tones;
the background stays flat.) Swap in
`birefnet-general` (~1 GB) for finer edges on hair and fur.

### Design with AI (relief generation)

This is the app's only image generator. Once an image is uploaded and selected, **Design
with AI** re-renders it as a monochrome sculpted relief — not a picture of a metal object,
but the same scene, same composition and aspect ratio, with every element modelled as clear
sculpted form, so the AI depth step turns it into a good heightmap:

- **What is it?** chips (Auto, Portrait, Animal, Landscape, Painting, Object, Logo/graphic):
  each tells the model how to sculpt that kind of image (a painting's brushstrokes become
  sculpted ridges, a landscape is layered from foreground to sky, …).
- **Style** chips (classic relief, deep sculpt, engraved, minimal, art deco) and a
  scrollable **Add to the design** gallery with a placeholder image per option
  (`static/relief/<id>.png`, drawn by `tools/make_relief_thumbs.py`): no background
  (first), background textures (stippled, brushed, sunburst, guilloché, hammered, stars),
  shading (deep, soft), an outline, and borders (beaded, laurel). One option per group
  applies. Picked choices show as highlighted tags in the chatbox.
- **Chatbox:** type a change ("my dog as a pirate"). A typed request makes **4
  variations**; choices alone make **1 image**. Results join the candidates (the first goes
  on the ring) and the height source switches to AI depth.
- **Every generation starts from the original upload**, even when a result is selected
  (re-generating from results drifted further from the original each time), and results
  are cropped to the original's exact aspect ratio.
- **Faithfulness** slider and a **usage** line (images and dollars today and all time;
  every image is logged to `data/usage.jsonl` with its model and cost).

The prompt (`app/relief_prompts.py`) is short and direct, because editing models follow
concise instructions and drop what comes late: "Convert this image into a monochrome
<style> sculpture rendering in smooth matte grey clay…", then the picked options and your
text, then the image-type guidance, faithfulness ("…apart from the requested changes", so
it never forbids them) and rules. It never says coin, medal or portrait — an editing model
takes those literally (it put Starry Night on a coin with an invented man). Styles, image
types and options are data in that file.

**Casting-friendly likeness.** Every prompt asks the model to keep the likeness but simplify
for casting: wrinkles, pores, stubble, individual hairs and fabric texture become smooth
broad planes, and hair and fur become a few bold, chunky locks. The Portrait and Animal
types add that the person or pet must stay recognisable to someone who knows them.

**Background options are flat, built by the app.** When a background option is picked (No
background or a texture), the model is asked only for the subject on a plain grey
background; the app then cuts the subject out (`background.subject_mask`, saved as
`subject_mask.png`) and lays it on a background it draws itself, so the background never has
shadows, depth or lighting:

- *No background*: the background is exactly 0 in the heightmap.
- *A texture*: the background is the texture's pattern, raised slightly —
  `BACKGROUND_TEXTURE_HEIGHT` (0.12 of the relief height, `app/config.py`) — with the
  subject sitting above it from `SUBJECT_BASE_LEVEL`.
- Without a background option, nothing changes: the whole image is sculpted as usual.

**Changing the background afterwards.** The **Background** chips under the crop image
(As made, No background, and each texture) work on any selected candidate, AI result or
upload, at any time: the subject is cut out of the image as it was made (kept as
`generated.png`; the solid cut-out mask is cached as `cutout_mask.png`), pasted on the chosen
background asset, and the cached depth is dropped, so depth → heightmap re-runs on the new
image. No new AI generation is needed. "As made" restores the original image.
`POST /api/designs/<id>/background` with `{"choice": "<background option id>" | null}`.

**Keeping the original.** At the default "close" faithfulness, the sentence straight after
the main instruction tells the model to keep everything in the original — the same framing,
the whole figure as far as it is visible (never a bust or close-up), every object — removing
and adding nothing apart from the requested changes.

**Where the textures live:** `static/textures/` — one grayscale PNG per texture (white =
raised, black = the flat background) and `textures.json` listing them as `{"id", "label",
"file", "mode", "tile_mm"}`. `"tile"` textures repeat every `tile_mm` millimetres (the same
density on every ring); `"cover"` textures stretch across the whole design area (sunburst,
guilloché). The built-in six are drawn by `tools/make_textures.py` (rerun it to change them;
it keeps any entries you add). `app/textures.py` loads, tiles and scales them, and draws the
flat preview used in the candidate image. Each gallery option points at its texture via
`Addition.texture` in `app/relief_prompts.py`. Keep features at least ~0.3 mm wide on the
ring, or the minimum-feature step (0.25 mm) erases them. To add your own for now: drop a PNG
in `static/textures/`, add an entry to `textures.json` and an `Addition` with that texture id.

Generation (`app/relief.py`) uses, in order of preference:

- **fal.ai** (`FAL_KEY` in `.env`): instruction-following image-editing models. Single images
  use `FAL_MODEL`, the 4-variation requests `FAL_MODEL_VARIATIONS`; both default to FLUX.1
  Kontext [dev] (~$0.025). Kontext [pro] (`fal-ai/flux-pro/kontext`, ~$0.04) is a little more
  faithful. The panel's **Model** chips (FLUX.2 / Kontext [pro], `FAL_MODEL_CHOICES` in
  `app/config.py`) switch between models for comparing them: the picked model makes both
  single images and variations, and the button shows its price. Qwen Image Edit (Plus) and
  FLUX.2 edit also work. The prompt is phrased as an instruction ("Convert this image
  into …"), with the faithfulness slider spelled out in words, since these models have no
  structure-strength setting. The source goes up as a ~1 MP JPEG data URI; one image per
  call, requests in parallel. Costs are fal's list prices per model (`fal_price_usd`).
- **Stability AI** (`STABILITY_API_KEY`, used when there's no fal key): the Structure Control
  endpoint, where the slider is the control strength; 5 credits ($0.05) per image.
- Otherwise a free offline stand-in, so the flow still works.

`RELIEF_PROVIDER` (fal / stability / mock) overrides the choice. Endpoints: `GET /api/relief/options`, `POST /api/relief`
(`source_id`, `style`, `image_type`, `additions`, `text`, `fidelity`, `model`), `GET /api/usage`.

### Image enhancement

Ring-agnostic tools under **Image enhancement** in the panel. They live in
`app/processing.py` (pure functions on heightmap-sized arrays) and `app/enhance.py` (AI
models), know nothing about which ring the design is for, and apply to every design ring,
so a new ring style gets them automatically. `process_image` runs them in this order:

1. **Clean up image** (`denoise`, on by default): two edge-preserving bilateral passes,
   so grain and compression noise don't become bumpy metal.
2. **AI upscale 2x** (`upscale`): Real-ESRGAN x2 on the full image, in tiles, cached as
   `upscaled.png` (~10 s on first use per image). Sharper source, smoother edges.
3. **Height from: AI depth** (`height_source = "depth"`): Depth Anything V2 Small
   estimates the photo's 3D shape (cached as `depth.png`, ~0.5 s), so height follows form
   (a nose, a cheek) instead of brightness (where dark hair or shadows would sink).
   **Fine detail** (`depth_detail`) blends the photo's texture finer than 0.4mm back on top.
4. **Bas-relief compression** (`bas_relief`): as for coins and medals. Gradients steeper
   than 3x the average are compressed in the gradient domain and the heightfield is
   rebuilt with a Poisson solve; then shapes broader than 1.5mm shrink most, mid-scale
   ones less, and detail under 0.3mm keeps its full height (`BAS_RELIEF_SCALES_MM`).
5. **Remove background** (see below).
6. **Engraved outlines** (`outline_strength`): XDoG edges of the image cut as grooves, and
   **Hatching** (`hatch_strength`, `hatch_spacing_mm`, `hatch_angle_deg`): parallel lines
   whose width follows the image's darkness, banknote style. Both cut at most
   `ENGRAVE_MAX_DEPTH` (half the relief height).
7. Quantize and minimum feature size, as before.
8. **Smooth edges** (`smooth_mm`, 0.04mm by default): a final light blur that turns
   stair-stepped walls between heights into short slopes, so edges in the metal look
   smooth rather than jagged.

The two models (ONNX, run on the CPU with onnxruntime, no PyTorch) download once to
`~/.cache/silversignal/models` (about 100MB + 67MB; `SILVERSIGNAL_MODELS_DIR` overrides).
The UI defaults to brightness with cleanup and edge smoothing on; for photos, try AI depth
with fine detail ~0.5 and bas-relief ~0.6.

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

## Prompt lab

`/lab.html` (linked from the designer's panel) tunes the AI relief prompt on a fixed
set of test images before any LoRA training data is made (`app/prompt_lab.py`):

1. **Test images**: upload images by category (people, animals, scenery, paintings,
   logos, drawings/cartoons, objects), each in the `tune` or `holdout` split (auto: every
   4th held out), with where it came from and its licence.
2. **Prompt versions**: `v1` is the app's built-in prompt. Copy a version, edit its
   settings (model, style, options, faithfulness) and any of its prompt text, and save.
   A version that has been run is locked, so a run always matches its version.
3. **Runs & review**: run a version over a split/categories. Each source keeps the same
   seed in every run, so differences come from the prompt. Each result shows the source,
   the generation, a shaded preview of its depth and the depth map; review it with a
   verdict (G/U/B), a 1–5 score, error tags and a comment, and compare it with another run.

Everything is plain files under `PROMPT_LAB_DIR` (default `data/prompt_lab/`, git-ignored),
described by the `README.md` the lab writes there: `sources/sources.csv`, `prompts/vN.json`,
`runs/<run>/<source>/{generation.webp, depth.png, relief.webp, meta.json, review.json}` and
a `summary.json` per run. A new prompt version can also be written straight to
`prompts/vN.json` (e.g. by Claude after reading the reviews) and run from the page.

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
- The report shows the **total ring weight** in 935 silver (93.5% silver, the rest copper:
  `SILVER_DENSITY_G_CM3` = 10.37 g/cm³ in `app/config.py`). The viewer computes it from the
  volume of the ring on show (its mesh is watertight; signed-tetrahedron sum,
  `meshVolume` in `static/main.js`) with the design applied at its true height — whatever
  the preview exaggeration — including the raised floor and moved
  ridge wall on Square (ridged). E.g. S Square with a small emblem is about 17.95 g. The
  report also still gives the relief's own volume and weight (same density).

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
- The ring STL's relief detail is limited by the ring model's mesh density in the design area
  (the 16-bit heightmap holds finer detail).

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
