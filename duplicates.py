"""Duplicate-image detection for DiamondVision Worker 3.0."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)

DEFAULT_HASH_SIZE = 16
DEFAULT_THRESHOLD = 18


@dataclass(frozen=True)
class DuplicateMatch:
    """Details about a duplicate comparison result."""

    is_duplicate: bool
    distance: int
    matched_identifier: str = ""


def _validate_image(image: np.ndarray) -> None:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Cannot hash an empty image.")

    if image.ndim not in {2, 3}:
        raise ValueError(
            f"Unsupported image shape for duplicate detection: {image.shape}"
        )


def average_hash(
    image: np.ndarray,
    hash_size: int = DEFAULT_HASH_SIZE,
) -> int:
    """Return an integer average hash for an OpenCV image."""
    _validate_image(image)

    if hash_size < 2:
        raise ValueError("hash_size must be at least 2.")

    if image.ndim == 3:
        if image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    resized = cv2.resize(
        gray,
        (hash_size, hash_size),
        interpolation=cv2.INTER_AREA,
    )

    mean_value = float(resized.mean())
    bits = resized >= mean_value

    hash_value = 0
    for bit in bits.flatten():
        hash_value = (hash_value << 1) | int(bool(bit))

    return hash_value


def hamming_distance(
    left_hash: int,
    right_hash: int,
) -> int:
    """Return the number of differing bits between two integer hashes."""
    return int((int(left_hash) ^ int(right_hash)).bit_count())


class DuplicateDetector:
    """Stateful duplicate detector using average hashes."""

    def __init__(
        self,
        threshold: int = DEFAULT_THRESHOLD,
        hash_size: int = DEFAULT_HASH_SIZE,
    ) -> None:
        if threshold < 0:
            raise ValueError("threshold cannot be negative.")

        if hash_size < 2:
            raise ValueError("hash_size must be at least 2.")

        self.threshold = int(threshold)
        self.hash_size = int(hash_size)
        self._entries: list[tuple[int, str]] = []

    def clear(self) -> None:
        """Remove all previously seen hashes."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def find_match(
        self,
        image: np.ndarray,
    ) -> DuplicateMatch:
        """Compare an image against all previously registered images."""
        current_hash = average_hash(
            image,
            hash_size=self.hash_size,
        )

        best_distance: int | None = None
        best_identifier = ""

        for stored_hash, identifier in self._entries:
            distance = hamming_distance(
                current_hash,
                stored_hash,
            )

            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_identifier = identifier

            if distance <= self.threshold:
                return DuplicateMatch(
                    is_duplicate=True,
                    distance=distance,
                    matched_identifier=identifier,
                )

        return DuplicateMatch(
            is_duplicate=False,
            distance=best_distance if best_distance is not None else -1,
            matched_identifier=best_identifier,
        )

    def register(
        self,
        image: np.ndarray,
        identifier: str = "",
    ) -> int:
        """Store an image hash and return the computed hash value."""
        image_hash = average_hash(
            image,
            hash_size=self.hash_size,
        )

        self._entries.append(
            (
                image_hash,
                str(identifier or ""),
            )
        )

        return image_hash

    def check(
        self,
        image: np.ndarray,
        identifier: str = "",
        register_unique: bool = True,
    ) -> bool:
        """Return True when duplicate; optionally register unique images."""
        match = self.find_match(image)

        if match.is_duplicate:
            LOGGER.debug(
                "Duplicate detected: identifier=%s matched=%s distance=%s",
                identifier,
                match.matched_identifier,
                match.distance,
            )
            return True

        if register_unique:
            self.register(
                image,
                identifier=identifier,
            )

        return False

    def is_duplicate(
        self,
        image: np.ndarray,
        identifier: str = "",
    ) -> bool:
        """Processor-facing API: check and register unique images."""
        return self.check(
            image,
            identifier=identifier,
            register_unique=True,
        )

    def add(
        self,
        image: np.ndarray,
        identifier: str = "",
    ) -> DuplicateMatch:
        """Check an image, register it when unique, and return match details."""
        match = self.find_match(image)

        if not match.is_duplicate:
            self.register(
                image,
                identifier=identifier,
            )

        return match

    def export_entries(self) -> list[dict[str, Any]]:
        """Return a serializable snapshot of stored hashes."""
        return [
            {
                "hash": image_hash,
                "identifier": identifier,
            }
            for image_hash, identifier in self._entries
        ]
