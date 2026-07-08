"""Cheap, dependency-free frame-change detection.

The playthrough loop pays a slow VLM round-trip for every executed action. Many
frames don't need a fresh decision, though — advancing dialogue is a run of
near-identical clicks, and a click that lands on nothing leaves the screen
unchanged. This module provides a microsecond-cost way to tell "did the screen
change, and how much?" so the harness can skip VLM calls it doesn't need while
never acting blind: every speculative action is gated by a real observation of
the resulting frame.

Pillow-only by design — the project ships Pillow but not numpy/imagehash. A
signature is a tiny grayscale thumbnail; a delta is the mean absolute pixel
difference between two signatures, normalised to 0.0–1.0.
"""

from __future__ import annotations

from PIL import Image, ImageChops, ImageStat

# Default thumbnail edge (px). Small enough to be free, large enough that a
# text/dialogue change registers while noise does not.
_SIGNATURE_SIZE = 32


def frame_signature(img: Image.Image, size: int = _SIGNATURE_SIZE) -> Image.Image:
    """Reduce *img* to a small grayscale thumbnail for cheap comparison.

    Downscaling to a fixed size makes the comparison resolution-independent and
    collapses sub-pixel noise, while grayscale keeps it insensitive to minor
    colour jitter. The result is layout-sensitive: a new screen or a moved
    element produces a large delta, whereas a couple of changed dialogue
    characters produce a small one.
    """
    return img.convert("L").resize((size, size))


def frame_delta(sig_a: Image.Image, sig_b: Image.Image) -> float:
    """Return the normalised mean absolute difference between two signatures.

    0.0 = pixel-identical, 1.0 = maximally different. Computed as the mean of
    ``|a - b|`` over all pixels, divided by 255.
    """
    diff = ImageChops.difference(sig_a, sig_b)
    mean = ImageStat.Stat(diff).mean[0]
    return mean / 255.0


def classify_change(
    delta: float, min_delta: float, structural_delta: float
) -> str:
    """Bucket a *delta* into ``"none"`` / ``"minor"`` / ``"structural"``.

    - ``"none"``       — below ``min_delta``; frames are effectively identical
                         (a click that did nothing, a settled screen).
    - ``"minor"``      — a text/dialogue change with stable layout (dialogue is
                         still rolling; safe to keep advancing).
    - ``"structural"`` — at or above ``structural_delta``; a new screen, choice,
                         or scene — a genuine decision point that warrants a
                         fresh VLM look.
    """
    if delta < min_delta:
        return "none"
    if delta >= structural_delta:
        return "structural"
    return "minor"
