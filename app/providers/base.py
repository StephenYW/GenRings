"""
Provider interface for text -> image generation.

Implementations only need to produce RGB images from a fully-formed prompt
string; everything about ring geometry / heightmap processing lives in
processing.py and is provider-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class ImageProvider(ABC):
    name: str = "base"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        n: int,
        seed: Optional[int] = None,
    ) -> list[np.ndarray]:
        """
        Generate `n` RGB uint8 images of shape (height, width, 3).

        `seed` (if given) should make output deterministic/reproducible
        where the underlying model supports it.
        """
        raise NotImplementedError
