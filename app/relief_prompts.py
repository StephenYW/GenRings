"""
Prompt building for AI relief generation (image -> sculpted metal relief).

The prompt is built in layers so the result always suits casting, whatever
the user adds:

  1. a locked base: re-render the source as a single-material polished silver
     bas-relief, lit flat from the front, smooth castable forms, no colour or
     text -- and stay close to the source's subject, pose and features;
  2. a style preset (classic coin, deep sculpt, ...);
  3. any additions picked from the gallery (background textures, shading,
     borders); within a group (e.g. background) only one applies;
  4. the user's own words, if any, placed as the subject description.

Additions and styles are data: add an entry here and it shows up in the
panel, with its placeholder image at static/relief/<id>.png
(tools/make_relief_thumbs.py draws them).
"""
from __future__ import annotations

from dataclasses import dataclass

BASE = (
    "a polished sterling silver bas-relief medallion of {subject}, sculpted solid metal relief, "
    "one single material, monochrome silver, frontal soft even studio lighting, smooth continuous "
    "sculpted forms with clear depth, crisp clean silhouette, clear separation between subject and "
    "background, simplified shapes that can be cast in metal, faithful to the original's subject, "
    "pose, proportions and features"
)
DEFAULT_SUBJECT = "the subject of this image"
NEGATIVE = (
    "color, colorful, photograph, photorealistic, skin texture, film grain, noise, text, letters, "
    "words, watermark, signature, logo, frame, cluttered, messy, tiny fragile details, blurry, "
    "low quality, deformed, distorted proportions, extra limbs, cartoon, flat 2d drawing"
)


@dataclass(frozen=True)
class Style:
    id: str
    label: str
    prompt: str


@dataclass(frozen=True)
class Addition:
    id: str
    label: str
    group: str   # one addition per group applies (the last one picked)
    prompt: str


STYLES: dict[str, Style] = {s.id: s for s in [
    Style("classic", "Classic coin", "classic coin portrait relief, balanced low relief, refined detail"),
    Style("deep", "Deep sculpt", "high relief sculpture with deep pronounced forms and strong volume"),
    Style("engraved", "Engraved", "fine engraved line work and crisp incised details, like a hand-engraved medal"),
    Style("minimal", "Minimal", "minimalist simplified forms, few large smooth planes, elegant and clean"),
    Style("art_deco", "Art deco", "art deco relief style, stylised geometric forms and streamlined planes"),
]}
DEFAULT_STYLE = "classic"

ADDITIONS: dict[str, Addition] = {a.id: a for a in [
    Addition("no_background", "No background", "background",
             "the subject isolated on a perfectly plain, flat, smooth empty background"),
    Addition("stippled", "Stippled", "background", "a finely stippled dotted texture covering the background"),
    Addition("brushed", "Brushed lines", "background", "fine parallel brushed lines across the background"),
    Addition("sunburst", "Sunburst", "background", "radiating sunburst rays fanning out behind the subject"),
    Addition("guilloche", "Guilloché", "background",
             "intricate guilloché wave engraving in the background, like a banknote"),
    Addition("hammered", "Hammered", "background", "a hand-hammered dimpled metal texture in the background"),
    Addition("stars", "Stars", "background", "small raised stars scattered across the background"),
    Addition("deep_shading", "Deep shading", "shading",
             "dramatic deep sculpted shading with strong recesses and high points"),
    Addition("soft_shading", "Soft shading", "shading", "soft gentle shading and subtle low relief modelling"),
    Addition("outline", "Outline", "outline", "a crisp engraved outline tracing the subject's silhouette"),
    Addition("beaded_border", "Beaded border", "border", "a raised beaded rim running around the edge"),
    Addition("laurel", "Laurel wreath", "border", "a sculpted laurel wreath framing the subject"),
]}


def resolve_additions(ids: list[str]) -> list[Addition]:
    """Known additions in the order picked, keeping only the last one per group."""
    by_group: dict[str, Addition] = {}
    for i in ids:
        a = ADDITIONS.get(i)
        if a:
            by_group.pop(a.group, None)
            by_group[a.group] = a
    return list(by_group.values())


def fidelity_phrase(fidelity: float) -> str:
    """How much the result may depart from the source, in words -- for editing
    models that have no structure-strength setting."""
    if fidelity >= 0.75:
        return ("keep the exact composition, framing, pose, proportions and identity of the original image; "
                "change only the material and style")
    if fidelity >= 0.55:
        return "keep the subject clearly recognisable, with the same overall composition"
    return "feel free to reinterpret the composition creatively"


def build_relief_prompt(style_id: str, addition_ids: list[str], user_text: str = "",
                        instruction: bool = False, fidelity: float = 0.8) -> tuple[str, str]:
    """(prompt, negative prompt) for one relief generation. `instruction`
    phrases it as an edit ("Transform this image into ...") for
    instruction-following editing models, with the faithfulness spelled out
    and the negatives folded in (most of them take no negative prompt)."""
    subject = user_text.strip() or DEFAULT_SUBJECT
    parts = [BASE.format(subject=subject), STYLES.get(style_id, STYLES[DEFAULT_STYLE]).prompt]
    parts += [a.prompt for a in resolve_additions(addition_ids)]
    if not instruction:
        return ", ".join(parts), NEGATIVE
    text = "Transform this image into " + ", ".join(parts) + ". " + fidelity_phrase(fidelity).capitalize() + "."
    text += " No color, no text or letters, no photographic texture, no frame."
    return text, NEGATIVE


def options() -> dict:
    """What the panel shows: styles and additions (in gallery order)."""
    return {
        "styles": [{"id": s.id, "label": s.label} for s in STYLES.values()],
        "default_style": DEFAULT_STYLE,
        "additions": [{"id": a.id, "label": a.label, "group": a.group, "thumb": f"/relief/{a.id}.png"}
                      for a in ADDITIONS.values()],
    }
