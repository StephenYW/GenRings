"""
Locked prompt templates per preset, plus each preset's default processing
params. The user's free-text prompt is only ever substituted into {prompt} —
never used to construct the rest of the instruction, so it can't escape the
"white/black, no text" constraints we rely on for processing.
"""
from __future__ import annotations

from dataclasses import dataclass

TEXT_KEYWORDS = (
    "text", "letters", "letter", "word", "words", "name", "initials",
    "font", "typography", "write", "written", "engrave the words",
    "monogram",
)


@dataclass(frozen=True)
class PresetConfig:
    id: str
    label: str
    template: str
    levels: int
    blur_mm: float


PRESETS: dict[str, PresetConfig] = {
    # Every template is written to produce images that survive the
    # processing pipeline (grayscale -> quantize -> morphological
    # open/close -> edge mask) with minimal quality loss:
    #   - a pure, perfectly flat black background (photographic backgrounds
    #     or soft vignettes get picked up as low-level "raised" noise)
    #   - a single centered subject, since the design isn't recentered
    #   - thick, simple, smooth-edged shapes well above MIN_FEATURE_MM, so
    #     the min-feature morphology step doesn't eat fine detail
    #   - explicitly non-photographic rendering, since photo textures
    #     (skin pores, fabric weave, film grain) all read as unwanted
    #     high-frequency relief noise once converted to a heightmap
    "emblem": PresetConfig(
        id="emblem",
        label="Emblem (tiered relief)",
        template=(
            "bold engraved emblem of {prompt}, flat 2D vector illustration, pure solid "
            "white silhouette shapes on a completely flat solid black background, no "
            "gradients, no drop shadow, no photographic texture, no noise, crisp clean "
            "smooth edges, high contrast, thick simple shapes, generous spacing between "
            "shapes, single centered subject, symmetrical composition, no text, no "
            "letters, no watermark, no signature, no border, no frame"
        ),
        levels=4,
        blur_mm=0.0,
    ),
    "lineart": PresetConfig(
        id="lineart",
        label="Line Art (2-level)",
        template=(
            "clean bold line-art engraving of {prompt}, flat 2D vector illustration, "
            "pure solid white outlines and strokes on a completely flat solid black "
            "background, uniform thick line weight well-separated from other lines, no "
            "gradients, no shading, no photographic texture, no noise, no hairline "
            "details, high contrast, single centered subject, no text, no letters, no "
            "watermark, no signature, no border, no frame"
        ),
        levels=2,
        blur_mm=0.0,
    ),
    "organic": PresetConfig(
        id="organic",
        label="Organic (continuous bas-relief)",
        template=(
            "flat 2D grayscale digital heightmap texture of {prompt}, top-down "
            "orthographic view, directly overhead, zero perspective, zero camera angle. "
            "This is a data visualization, not a photo and not a 3D render: pixel "
            "brightness directly encodes elevation, white is highest and black is "
            "lowest, like a terrain heightmap used in a game engine or a digital "
            "elevation model. Flat even brightness with no directional lighting, no "
            "shadows, no highlights, no specular reflections, no ambient occlusion, no "
            "rim light. Smooth continuous grayscale gradients, no fine surface noise, "
            "no film grain, no photographic texture. Single centered composition with "
            "generous flat solid black empty space inset on all four sides — the "
            "subject does not touch or extend to any edge of the frame. No decorative "
            "border, no frame graphic, no vignette, no circular medallion or disc "
            "shape drawn around it, no text, no letters, no watermark, no signature"
        ),
        levels=0,
        blur_mm=0.15,
    ),
}

DEFAULT_PRESET = "emblem"


def build_prompt(user_prompt: str, preset_id: str) -> str:
    preset = PRESETS.get(preset_id, PRESETS[DEFAULT_PRESET])
    return preset.template.format(prompt=user_prompt.strip())


def mentions_text(user_prompt: str) -> bool:
    low = user_prompt.lower()
    return any(kw in low for kw in TEXT_KEYWORDS)
