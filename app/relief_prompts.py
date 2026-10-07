"""
Prompt building for AI relief generation.

The goal is NOT a picture of a metal object: it's the same image, re-rendered
so AI depth estimation turns it into a good heightmap -- one material, every
element modelled as clear sculpted form with readable depth, lit softly from
the front, no colour, no clutter. So the prompt never says "coin", "medal" or
"portrait" (an editing model takes those literally: it puts the scene on a
coin, or invents a portrait when there's no person). It's kept short and
direct, because editing models (FLUX Kontext etc.) follow concise
instructions and drop what comes late in a long prompt.

Layers, in order:
  1. the conversion instruction (with the style's relief character);
  2. what kind of image it is (portrait, animal, landscape, painting, ...),
     so each is sculpted in the way that suits it;
  3. the options picked from the gallery (background, shading, outline,
     border), one per group, as direct instructions;
  4. the user's own words, as a change to make;
  5. faithfulness (keep composition/framing/aspect) and the hard rules.

Styles, image types and options are data: add an entry here and it shows up
in the panel (options also need a placeholder image, static/relief/<id>.png,
drawn by tools/make_relief_thumbs.py).
"""
from __future__ import annotations

from dataclasses import dataclass

CONVERT = "Convert this image into a monochrome {style} sculpture rendering in smooth matte grey clay"
DESCRIBE = ("monochrome {style} sculpture rendering in smooth matte grey clay of the same scene as the source "
            "image, every element sculpted with clear depth and volume, soft even frontal lighting")
# Casting-friendly simplification, for every image: keep the likeness, lose
# the photographic fine detail that reads as noise in metal.
CASTING = ("Simplify it for metal casting while keeping the likeness: replace fine detail (wrinkles, pores, "
           "stubble, individual hairs, fabric texture) with smooth broad planes, and sculpt hair and fur as a "
           "few bold, chunky grouped locks.")
# With a background option, the model draws only the subject on a plain flat
# background; the app cuts it out and builds the background itself (flat,
# shadow-free, the chosen texture raised slightly -- see app/textures.py).
SUBJECT_ONLY = ("Show only the main subject, cleanly separated, on a perfectly plain, flat, uniform light grey "
                "background: no texture, no gradient, no lighting or shading on the background, and no shadow "
                "cast by the subject.")
# Straight after the main instruction (where editing models weigh it most):
# the result must match the original -- models like to crop people into a
# bust or close-up, drop things, or invent new ones.
KEEP_ALL = ("Keep everything in the original: the same framing, the whole figure as far as it is visible (never "
            "cropped into a bust or a close-up), and every object and background element -- remove nothing and "
            "add nothing{except_changes}.")
KEEP_SUBJECT = ("Keep the subject exactly as in the original: the whole figure as far as it is visible (never "
                "cropped into a bust or a close-up), at the same size and position -- remove nothing from it and "
                "add nothing{except_changes}.")
EXCEPT_CHANGES = " apart from the requested changes"
RULES = ("Clear readable depth (near parts raised, far parts recessed), soft even frontal light, one material, "
         "no colour, no text. Do not add a coin, medal, frame or border unless asked, or add or remove subjects.")
NEGATIVE = ("color, text, letters, watermark, signature, coin, medal, medallion, bust, frame, border, extra people, "
            "extra subjects, photograph, noise, grain, blurry, low quality, deformed, cluttered")


@dataclass(frozen=True)
class Style:
    id: str
    label: str
    relief: str     # goes into CONVERT as {style}
    detail: str     # an extra sentence


@dataclass(frozen=True)
class ImageType:
    id: str
    label: str
    prompt: str


@dataclass(frozen=True)
class Addition:
    id: str
    label: str
    group: str      # one addition per group applies (the last one picked)
    prompt: str
    texture: str | None = None  # background options: the texture the app lays behind the subject


STYLES: dict[str, Style] = {s.id: s for s in [
    Style("classic", "Classic relief", "low-relief", "Refined, balanced detail with gentle depth."),
    Style("deep", "Deep sculpt", "high-relief", "Bold, pronounced forms with strong depth and volume."),
    Style("engraved", "Engraved", "carved low-relief", "Crisp carved lines and incised detail."),
    Style("minimal", "Minimal", "simplified low-relief", "Few large smooth planes, minimal fine detail."),
    Style("art_deco", "Art deco", "art deco relief", "Stylised geometric forms and streamlined planes."),
]}
DEFAULT_STYLE = "classic"

IMAGE_TYPES: dict[str, ImageType] = {t.id: t for t in [
    ImageType("auto", "Auto", "Sculpt whatever the image shows -- people, animals, objects, landscapes or "
              "abstract shapes -- as it is."),
    ImageType("portrait", "Portrait", "It is a portrait of a real person who must stay instantly recognisable: "
              "keep their exact face shape, features, proportions, expression and hairstyle."),
    ImageType("animal", "Animal", "It shows an animal (perhaps a pet) that its owner must recognise: keep its "
              "exact breed, proportions, markings and expression."),
    ImageType("landscape", "Landscape", "It is a landscape: build depth in layers from foreground to "
              "background -- near features raised highest, the sky and distance lowest."),
    ImageType("painting", "Painting", "It is a painting: keep its composition and every element, and turn its "
              "brushstrokes, swirls and textures into sculpted ridges and grooves."),
    ImageType("drawing", "Drawing / cartoon", "It is a drawing or cartoon: keep its exact line work, character "
              "design and proportions; turn its outlines into crisp raised edges and its flat colour areas into "
              "smooth plateaus at a few clear heights."),
    ImageType("object", "Object", "It shows an object: sculpt its shape and surface details with crisp "
              "edges and correct proportions."),
    ImageType("logo", "Logo / graphic", "It is a logo or graphic: make its shapes flat raised plateaus at "
              "two or three clear heights, with crisp edges."),
]}
DEFAULT_IMAGE_TYPE = "auto"

ADDITIONS: dict[str, Addition] = {a.id: a for a in [
    # Background options: the model draws the subject alone (SUBJECT_ONLY) and
    # the app adds the flat background, textured with static/textures/<texture>.png.
    Addition("no_background", "No background", "background", SUBJECT_ONLY),
    Addition("stippled", "Stippled", "background", SUBJECT_ONLY, "stippled"),
    Addition("brushed", "Brushed lines", "background", SUBJECT_ONLY, "brushed"),
    Addition("sunburst", "Sunburst", "background", SUBJECT_ONLY, "sunburst"),
    Addition("guilloche", "Guilloché", "background", SUBJECT_ONLY, "guilloche"),
    Addition("hammered", "Hammered", "background", SUBJECT_ONLY, "hammered"),
    Addition("stars", "Stars", "background", SUBJECT_ONLY, "stars"),
    Addition("deep_shading", "Deep shading", "shading",
             "Exaggerate the depth: deep recesses and strongly raised high points."),
    Addition("soft_shading", "Soft shading", "shading", "Keep the depth gentle and shallow, with soft transitions."),
    Addition("outline", "Outline", "outline", "Add a crisp carved outline tracing the main subject's silhouette."),
    Addition("beaded_border", "Beaded border", "border", "Add a raised beaded border running around the edge of the image."),
    Addition("laurel", "Laurel wreath", "border", "Add a sculpted laurel wreath framing the main subject."),
]}


def background_choice(ids: list[str]) -> Addition | None:
    """The background option picked, if any (no background, or a texture)."""
    return next((a for a in resolve_additions(ids) if a.group == "background"), None)


def resolve_additions(ids: list[str]) -> list[Addition]:
    """Known additions in the order picked, keeping only the last one per group."""
    by_group: dict[str, Addition] = {}
    for i in ids:
        a = ADDITIONS.get(i)
        if a:
            by_group.pop(a.group, None)
            by_group[a.group] = a
    return list(by_group.values())


def fidelity_phrase(fidelity: float, has_changes: bool = False) -> str:
    """How much the result may depart from the source, in words -- for editing
    models that have no structure-strength setting. With requested changes
    (options or the user's text) it must not forbid them, or the model keeps
    the original background etc. and ignores the request."""
    except_changes = EXCEPT_CHANGES if has_changes else ""
    if fidelity >= 0.75:
        return (f"Keep the exact composition, framing, proportions and positions of everything in the "
                f"original{except_changes}.")
    if fidelity >= 0.55:
        return (f"Keep the same overall composition (never cropped into a bust or a close-up) and keep "
                f"everything recognisable{except_changes}.")
    return "You may reinterpret the composition creatively."


def default_templates() -> dict:
    """Every piece of text a relief prompt is built from, as plain data: the
    prompt lab stores a copy per prompt version and edits it there (see
    app/prompt_lab.py), so wording can be tried without changing this file."""
    return {
        "convert": CONVERT, "describe": DESCRIBE, "keep_all": KEEP_ALL, "keep_subject": KEEP_SUBJECT,
        "except_changes": EXCEPT_CHANGES, "casting": CASTING, "rules": RULES, "negative": NEGATIVE,
        "image_types": {t.id: t.prompt for t in IMAGE_TYPES.values()},
        "styles": {s.id: {"relief": s.relief, "detail": s.detail} for s in STYLES.values()},
        "additions": {a.id: a.prompt for a in ADDITIONS.values()},
    }


def build_relief_prompt(style_id: str, addition_ids: list[str], user_text: str = "",
                        instruction: bool = False, fidelity: float = 0.8,
                        image_type: str = DEFAULT_IMAGE_TYPE, templates: dict | None = None) -> tuple[str, str]:
    """(prompt, negative prompt) for one relief generation. `instruction`
    phrases it as a direct edit for instruction-following editing models
    (fal: FLUX Kontext, Qwen Image Edit, FLUX.2); otherwise it's a
    description, for Stable Diffusion with structure control (Stability).
    `templates` replaces any of default_templates()' text (a prompt lab version)."""
    t = default_templates()
    for key, value in (templates or {}).items():
        t[key] = {**t[key], **value} if isinstance(t.get(key), dict) else value
    style_id = style_id if style_id in t["styles"] else DEFAULT_STYLE
    style = t["styles"][style_id]
    kind = t["image_types"].get(image_type, t["image_types"][DEFAULT_IMAGE_TYPE])
    resolved = resolve_additions(addition_ids)
    adds = [t["additions"].get(a.id, a.prompt) for a in resolved]
    text = user_text.strip()
    if not instruction:
        parts = [t["describe"].format(style=style["relief"]), style["detail"], kind, t["casting"], *adds]
        if text:
            parts.insert(1, text)
        return " ".join(parts), t["negative"]
    # the requested changes come straight after the main instruction, where
    # editing models weigh them most
    bg = background_choice(addition_ids)
    convert = t["convert"].format(style=style["relief"]) + ("." if bg else ", showing exactly the same scene.")
    # a background option is how the subject is shown, not a change to it
    changes = bool(text) or any(a.group != "background" for a in resolved)
    except_changes = t["except_changes"] if changes else ""
    # At "close" faithfulness, keeping everything comes first (and replaces
    # the faithfulness sentence); lower settings use the looser wording late.
    close = fidelity >= 0.75
    parts = [convert]
    if close:
        parts.append((t["keep_subject"] if bg else t["keep_all"]).format(except_changes=except_changes))
    parts += adds
    if text:
        parts.append(f"Also make this change: {text}.")
    parts += [kind, style["detail"], t["casting"]]
    if not close:
        parts.append(fidelity_phrase(fidelity, bool(adds or text)))
    parts.append(t["rules"])
    return " ".join(p for p in parts if p), t["negative"]


def options() -> dict:
    """What the panel shows: styles, image types and additions (in gallery order)."""
    return {
        "styles": [{"id": s.id, "label": s.label} for s in STYLES.values()],
        "default_style": DEFAULT_STYLE,
        "image_types": [{"id": t.id, "label": t.label} for t in IMAGE_TYPES.values()],
        "default_image_type": DEFAULT_IMAGE_TYPE,
        "additions": [{"id": a.id, "label": a.label, "group": a.group, "thumb": f"/relief/{a.id}.png"}
                      for a in ADDITIONS.values()],
    }
