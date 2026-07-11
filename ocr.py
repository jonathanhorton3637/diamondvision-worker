"""OCR and roster utilities for DiamondVision Worker 3.0."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import threading
from typing import Any, Mapping

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

try:
    import easyocr
except ImportError:  # pragma: no cover - handled at runtime
    easyocr = None

_READER: Any | None = None
_READER_LOCK = threading.Lock()


@dataclass(frozen=True)
class OCRResult:
    """Result returned by jersey-number OCR."""

    number: str = ""
    confidence: float = 0.0
    raw: str = ""

    def as_tuple(self) -> tuple[str, float, str]:
        return self.number, self.confidence, self.raw


def easyocr_available() -> bool:
    """Return True when EasyOCR can be imported."""
    return easyocr is not None


def get_ocr_reader() -> Any | None:
    """Create the shared EasyOCR reader once and reuse it."""
    global _READER

    if easyocr is None:
        LOGGER.warning("EasyOCR is not installed; jersey OCR is disabled.")
        return None

    if _READER is not None:
        return _READER

    with _READER_LOCK:
        if _READER is not None:
            return _READER

        gpu_enabled = False
        try:
            import torch
            gpu_enabled = bool(torch.cuda.is_available())
        except Exception:
            LOGGER.debug("PyTorch CUDA detection failed.", exc_info=True)

        LOGGER.info("Initializing EasyOCR reader (gpu=%s).", gpu_enabled)
        _READER = easyocr.Reader(["en"], gpu=gpu_enabled, verbose=False)
        return _READER


def parse_roster_text(text: str | None) -> dict[str, str]:
    """
    Parse roster lines such as:
        12 Jane Smith
        #7, Taylor Jones
        18 - Morgan Lee
    """
    roster: dict[str, str] = {}

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = re.match(r"^\s*#?\s*(\d{1,3})\s*(?:[-,:|]\s*|\s+)(.+?)\s*$", line)
        if not match:
            LOGGER.debug("Skipping unrecognized roster line: %r", raw_line)
            continue

        number = normalize_jersey_number(match.group(1))
        name = re.sub(r"\s+", " ", match.group(2)).strip()

        if number and name:
            roster[number] = name

    return roster


def normalize_jersey_number(value: object) -> str:
    """Normalize a jersey number to one to three digits."""
    digits = re.sub(r"\D", "", str(value or ""))
    return digits[:3] if 1 <= len(digits) <= 3 else ""


def match_roster_number(
    detected_number: str | None,
    roster: Mapping[str, str],
) -> str:
    """Match an OCR result to a roster number using conservative corrections."""
    number = normalize_jersey_number(detected_number)
    if not number or not roster:
        return ""

    normalized_roster = {
        normalize_jersey_number(key): value
        for key, value in roster.items()
        if normalize_jersey_number(key)
    }

    if number in normalized_roster:
        return number

    doubled = number + number
    if doubled in normalized_roster:
        return doubled

    if len(number) == 2 and number[0] == number[1] and number[0] in normalized_roster:
        return number[0]

    contained = [
        candidate
        for candidate in normalized_roster
        if number in candidate or candidate in number
    ]
    if len(contained) == 1:
        return contained[0]

    # Resolve common OCR confusions only when the result becomes unique.
    translations = (
        str.maketrans({"0": "8"}),
        str.maketrans({"8": "0"}),
        str.maketrans({"1": "7"}),
        str.maketrans({"7": "1"}),
    )
    alternatives = {
        number.translate(table)
        for table in translations
        if number.translate(table) != number
    }
    matches = sorted(alternatives.intersection(normalized_roster))
    return matches[0] if len(matches) == 1 else ""


def roster_match_number(
    ocr_number: str | None,
    roster: Mapping[str, str],
) -> str:
    """Backward-compatible alias."""
    return match_roster_number(ocr_number, roster)


def jersey_crops(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Return multiple body regions likely to contain a jersey number."""
    if image is None or image.size == 0:
        return []

    height, width = image.shape[:2]
    definitions = (
        ("full_center", 0.08, 0.92, 0.12, 0.88),
        ("torso_center", 0.18, 0.82, 0.18, 0.82),
        ("chest_center", 0.18, 0.68, 0.24, 0.76),
        ("upper_wide", 0.05, 0.72, 0.08, 0.92),
        ("lower_torso", 0.34, 0.94, 0.16, 0.84),
        ("left_body", 0.10, 0.88, 0.02, 0.60),
        ("right_body", 0.10, 0.88, 0.40, 0.98),
    )

    crops: list[tuple[str, np.ndarray]] = []
    for name, top, bottom, left, right in definitions:
        crop = image[
            int(height * top):int(height * bottom),
            int(width * left):int(width * right),
        ]
        if crop is not None and crop.size and min(crop.shape[:2]) >= 40:
            crops.append((name, crop))

    return crops


def preprocess_for_ocr(crop: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Create contrast and threshold variants for numeric OCR."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, 7, 45, 45)

    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )
    _, otsu = cv2.threshold(
        denoised,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    variants = [
        ("gray", gray),
        ("clahe", enhanced),
        ("adaptive", adaptive),
        ("adaptive_inv", cv2.bitwise_not(adaptive)),
        ("otsu", otsu),
        ("otsu_inv", cv2.bitwise_not(otsu)),
    ]

    output: list[tuple[str, np.ndarray]] = []
    for name, variant in variants:
        height, width = variant.shape[:2]
        scale = 2 if max(height, width) >= 500 else 3
        output.append(
            (
                name,
                cv2.resize(
                    variant,
                    (width * scale, height * scale),
                    interpolation=cv2.INTER_CUBIC,
                ),
            )
        )

    return output


def _candidate_score(number: str, confidence: float, roster: Mapping[str, str]) -> float:
    score = confidence

    if len(number) <= 2:
        score += 0.05
    elif len(number) == 3:
        score -= 0.08

    if number in roster:
        score += 0.22
    elif match_roster_number(number, roster):
        score += 0.12

    return score


def read_jersey_number_result(
    image: np.ndarray,
    roster: Mapping[str, str] | None = None,
) -> OCRResult:
    """Read the most likely jersey number from an OpenCV BGR image."""
    reader = get_ocr_reader()
    if reader is None:
        return OCRResult(raw="OCR_UNAVAILABLE")

    active_roster = roster or {}
    candidates: dict[str, dict[str, Any]] = {}
    raw_hits: list[str] = []

    try:
        for crop_name, crop in jersey_crops(image):
            for version_name, processed in preprocess_for_ocr(crop):
                detections = reader.readtext(
                    processed,
                    detail=1,
                    paragraph=False,
                    allowlist="0123456789",
                    decoder="greedy",
                )

                for detection in detections:
                    if len(detection) < 3:
                        continue

                    number = normalize_jersey_number(detection[1])
                    confidence = float(detection[2])

                    if not number or confidence < 0.08:
                        continue
                    if len(number) == 3 and confidence < 0.60 and number not in active_roster:
                        continue

                    raw_hits.append(
                        f"{crop_name}/{version_name}:{number}:{confidence:.3f}"
                    )

                    item = candidates.setdefault(
                        number,
                        {"votes": 0, "best_confidence": 0.0, "score": 0.0},
                    )
                    item["votes"] += 1
                    item["best_confidence"] = max(item["best_confidence"], confidence)
                    item["score"] += _candidate_score(number, confidence, active_roster)

        if not candidates:
            return OCRResult(raw=" | ".join(raw_hits))

        def ranking(item: tuple[str, dict[str, Any]]) -> tuple[float, int, float]:
            _, stats = item
            average = stats["score"] / max(stats["votes"], 1)
            vote_bonus = min(stats["votes"], 8) * 0.025
            return average + vote_bonus, stats["votes"], stats["best_confidence"]

        best_number, best_stats = max(candidates.items(), key=ranking)
        matched = match_roster_number(best_number, active_roster)
        final_number = matched or best_number

        confidence = min(
            1.0,
            best_stats["best_confidence"] + min(best_stats["votes"] - 1, 5) * 0.03,
        )
        return OCRResult(final_number, confidence, " | ".join(raw_hits))

    except Exception as error:
        LOGGER.exception("Jersey OCR failed.")
        return OCRResult(raw=f"OCR_ERROR:{type(error).__name__}:{error}")


def read_jersey_number(
    image: np.ndarray,
    roster: Mapping[str, str] | None = None,
) -> tuple[str, float, str]:
    """Backward-compatible tuple API used by processor.py."""
    return read_jersey_number_result(image, roster).as_tuple()
