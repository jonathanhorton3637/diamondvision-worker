"""Photo-quality scoring for DiamondVision Worker 3.0."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PhotoScore:
    """Normalized quality metrics and final classification for one image."""

    sharpness: float
    exposure: float
    contrast: float
    center_detail: float
    score: float
    category: str

    @property
    def total(self) -> float:
        """Backward-compatible alias for the final score."""
        return self.score

    @property
    def total_score(self) -> float:
        """Backward-compatible alias for the final score."""
        return self.score

    @property
    def classification(self) -> str:
        """Backward-compatible alias for the category."""
        return self.category

    @classmethod
    def from_image(cls, image: np.ndarray) -> "PhotoScore":
        """Build a complete score from an OpenCV BGR image."""
        return score_image(image)


def _validate_image(image: np.ndarray) -> None:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Cannot score an empty image.")

    if image.ndim not in {2, 3}:
        raise ValueError(
            f"Unsupported image shape for scoring: {image.shape}"
        )


def _grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _normalize_metric(
    value: float,
    low: float,
    high: float,
) -> float:
    if high <= low:
        raise ValueError("Metric normalization range is invalid.")

    normalized = (value - low) / (high - low)
    return float(max(0.0, min(100.0, normalized * 100.0)))


def sharpness_score(image: np.ndarray) -> float:
    """Measure global edge detail using Laplacian variance."""
    _validate_image(image)
    gray = _grayscale(image)

    variance = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    # Log scaling prevents extremely sharp files from dominating.
    scaled = math.log1p(max(0.0, variance))
    return round(
        _normalize_metric(
            scaled,
            math.log1p(20.0),
            math.log1p(1600.0),
        ),
        3,
    )


def exposure_score(image: np.ndarray) -> float:
    """Score brightness balance while penalizing clipped shadows/highlights."""
    _validate_image(image)
    gray = _grayscale(image).astype(np.float32)

    mean_brightness = float(gray.mean())
    dark_fraction = float(np.mean(gray <= 8.0))
    bright_fraction = float(np.mean(gray >= 247.0))

    target = 128.0
    brightness_penalty = min(
        1.0,
        abs(mean_brightness - target) / target,
    )
    clipping_penalty = min(
        1.0,
        (dark_fraction + bright_fraction) * 2.75,
    )

    score = 100.0 * (
        1.0
        - (0.72 * brightness_penalty)
        - (0.28 * clipping_penalty)
    )

    return round(
        max(0.0, min(100.0, score)),
        3,
    )


def contrast_score(image: np.ndarray) -> float:
    """Measure tonal spread while avoiding rewards for extreme clipping."""
    _validate_image(image)
    gray = _grayscale(image)

    percentile_5 = float(np.percentile(gray, 5))
    percentile_95 = float(np.percentile(gray, 95))
    dynamic_range = percentile_95 - percentile_5

    return round(
        _normalize_metric(
            dynamic_range,
            28.0,
            190.0,
        ),
        3,
    )


def center_detail_score(image: np.ndarray) -> float:
    """Measure detail in the central subject region."""
    _validate_image(image)

    height, width = image.shape[:2]

    top = int(height * 0.20)
    bottom = int(height * 0.82)
    left = int(width * 0.18)
    right = int(width * 0.82)

    crop = image[top:bottom, left:right]

    if crop.size == 0:
        crop = image

    gray = _grayscale(crop)
    variance = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    scaled = math.log1p(max(0.0, variance))

    return round(
        _normalize_metric(
            scaled,
            math.log1p(18.0),
            math.log1p(1800.0),
        ),
        3,
    )


def classify(score: float) -> str:
    """Map a 0–100 score to the DiamondVision output category."""
    numeric = float(score)

    if numeric >= 70.0:
        return "Best"

    if numeric >= 38.0:
        return "Keep"

    return "Reject"


def score_image(image: np.ndarray) -> PhotoScore:
    """Calculate all quality metrics and the final weighted score."""
    _validate_image(image)

    sharpness = sharpness_score(image)
    exposure = exposure_score(image)
    contrast = contrast_score(image)
    center_detail = center_detail_score(image)

    total = (
        (sharpness * 0.38)
        + (center_detail * 0.32)
        + (exposure * 0.18)
        + (contrast * 0.12)
    )

    total = round(
        max(0.0, min(100.0, total)),
        3,
    )

    category = classify(total)

    LOGGER.debug(
        (
            "Photo score: total=%.3f category=%s "
            "sharpness=%.3f exposure=%.3f "
            "contrast=%.3f center_detail=%.3f"
        ),
        total,
        category,
        sharpness,
        exposure,
        contrast,
        center_detail,
    )

    return PhotoScore(
        sharpness=sharpness,
        exposure=exposure,
        contrast=contrast,
        center_detail=center_detail,
        score=total,
        category=category,
    )


def total_score(image: np.ndarray) -> float:
    """Backward-compatible numeric scoring API."""
    return score_image(image).score
