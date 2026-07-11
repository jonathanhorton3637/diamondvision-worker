"""Core DiamondVision image-processing pipeline.

This module intentionally contains no Dropbox, RunPod, ZIP, or temporary-directory code.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from duplicates import DuplicateDetector
from image_utils import copy_unique, find_images, save_display_jpeg
from ocr import match_roster_number, parse_roster_text, read_jersey_number
import scoring

LOGGER = logging.getLogger(__name__)
ProgressCallback = Callable[[int, int, str], None]


def safe_name(value: object) -> str:
    """Return a filesystem-safe folder component."""
    text = re.sub(r"\s+", "_", str(value or "").strip()) or "Unknown"
    text = re.sub(r'[<>:"/\\|?*#\x00-\x1f]', "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip(" ._") or "Unknown"


def _load_image(path: str) -> np.ndarray | None:
    """Call the image loader exposed by image_utils.py."""
    import image_utils

    for name in ("load_image", "load_image_fast", "load"):
        loader = getattr(image_utils, name, None)
        if callable(loader):
            return loader(path)

    raise AttributeError(
        "image_utils.py must expose load_image(path) or load_image_fast(path)."
    )


def _score_image(image: np.ndarray) -> tuple[float, str, dict[str, float]]:
    """Normalize supported scoring.py APIs to score/category/component values."""
    score_function = getattr(scoring, "score_image", None)
    result: Any

    if callable(score_function):
        result = score_function(image)
    else:
        photo_score_class = getattr(scoring, "PhotoScore", None)
        if photo_score_class and hasattr(photo_score_class, "from_image"):
            result = photo_score_class.from_image(image)
        elif callable(getattr(scoring, "total_score", None)):
            numeric = float(scoring.total_score(image))
            classify_function = getattr(scoring, "classify")
            return numeric, str(classify_function(numeric)), {}
        else:
            raise AttributeError(
                "scoring.py must expose score_image(image), "
                "PhotoScore.from_image(image), or total_score(image)."
            )

    if is_dataclass(result):
        payload = asdict(result)
    elif isinstance(result, Mapping):
        payload = dict(result)
    elif isinstance(result, (int, float)):
        payload = {"score": float(result)}
    else:
        payload = {
            key: getattr(result, key)
            for key in (
                "score",
                "total",
                "total_score",
                "category",
                "classification",
                "sharpness",
                "exposure",
                "contrast",
                "center_detail",
            )
            if hasattr(result, key)
        }

    numeric = float(
        payload.get("score", payload.get("total", payload.get("total_score", 0.0)))
    )

    category = payload.get("category", payload.get("classification"))
    if not category:
        classify_function = getattr(scoring, "classify", None)
        category = classify_function(numeric) if callable(classify_function) else (
            "Best" if numeric >= 70 else "Keep" if numeric >= 38 else "Reject"
        )

    components = {}
    for key in ("sharpness", "exposure", "contrast", "center_detail"):
        if key in payload:
            components[key] = round(float(payload[key]), 3)

    return numeric, str(category).title(), components


def _is_duplicate(detector: DuplicateDetector, image: np.ndarray, identifier: str) -> bool:
    """Support common DuplicateDetector interfaces."""
    for name in ("is_duplicate", "check"):
        method = getattr(detector, name, None)
        if callable(method):
            try:
                return bool(method(image, identifier))
            except TypeError:
                return bool(method(image))

    add_method = getattr(detector, "add", None)
    if callable(add_method):
        try:
            result = add_method(image, identifier)
        except TypeError:
            result = add_method(image)

        if isinstance(result, bool):
            return result
        if hasattr(result, "is_duplicate"):
            return bool(result.is_duplicate)

    raise AttributeError(
        "DuplicateDetector must expose is_duplicate(), check(), or add()."
    )


def _detect_color_name(image: np.ndarray) -> str:
    height, width = image.shape[:2]
    crop = image[
        int(height * 0.20):int(height * 0.80),
        int(width * 0.25):int(width * 0.75),
    ]
    if crop.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = (saturation > 35) & (value > 35)

    mean_s = float(saturation.mean())
    mean_v = float(value.mean())
    mean_h = float(hsv[:, :, 0][mask].mean()) if np.any(mask) else 0.0

    if mean_v > 185 and mean_s < 70:
        return "white"
    if mean_v < 75:
        return "black"
    if mean_s < 55:
        return "gray"
    if mean_h < 10 or mean_h >= 165:
        return "red"
    if 10 <= mean_h < 20:
        return "orange"
    if 20 <= mean_h < 36:
        return "yellow"
    if 36 <= mean_h < 88:
        return "green"
    if 88 <= mean_h < 132:
        return "blue"
    if 132 <= mean_h < 165:
        return "purple"
    return "unknown"


def detect_team_from_color(
    image: np.ndarray,
    team1_color: str = "",
    team2_color: str = "",
) -> str:
    """Return Team_1 or Team_2 using a conservative torso-color comparison."""
    color = _detect_color_name(image)
    team1 = str(team1_color or "").strip().lower()
    team2 = str(team2_color or "").strip().lower()

    if team2 and color == team2 and color != team1:
        return "Team_2"
    return "Team_1"


def _normalize_job_config(value: str | Mapping[str, Any] | None) -> dict[str, Any]:
    if isinstance(value, Mapping):
        config = dict(value)
    else:
        config = {
            "mode": "single",
            "team1": "Team",
            "team1_color": "",
            "roster1": str(value or ""),
            "team2": "Opponent",
            "team2_color": "",
            "roster2": "",
        }

    mode = str(config.get("mode", "single")).strip().lower()
    config["mode"] = "two_team" if mode in {"two_team", "two-team", "two", "2"} else "single"
    config.setdefault("team1", "Team")
    config.setdefault("team2", "Opponent")
    config.setdefault("team1_color", "")
    config.setdefault("team2_color", "")
    config.setdefault("roster1", "")
    config.setdefault("roster2", "")
    config.setdefault("ocr_min_confidence", 0.25)
    config.setdefault("best_of_tournament_limit", 75)
    config.setdefault("duplicate_threshold", 3)
    return config


def _create_output_folders(
    output_dir: Path,
    mode: str,
    team_names: tuple[str, str],
) -> None:
    base_folders = (
        "Originals",
        "Best",
        "Keep",
        "Reject",
        "Duplicates",
        "BestOfTournament",
        "Players/Unknown",
        "Favorites",
        "Reports",
    )
    for folder in base_folders:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    if mode == "two_team":
        for team in team_names:
            for folder in ("Best", "Keep", "Reject", "Duplicates", "Players/Unknown"):
                (output_dir / "Teams" / team / folder).mkdir(
                    parents=True,
                    exist_ok=True,
                )


def _relative(path: str | Path | None, root: Path) -> str:
    if not path:
        return ""
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return str(path)


def _make_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        "total": len(rows),
        "best": 0,
        "keep": 0,
        "reject": 0,
        "duplicates": 0,
        "matched": 0,
        "unknown": 0,
        "errors": 0,
    }

    for row in rows:
        category = str(row["Category"]).lower()
        if category == "best":
            counts["best"] += 1
        elif category == "keep":
            counts["keep"] += 1
        elif category == "reject":
            counts["reject"] += 1
        elif category == "duplicates":
            counts["duplicates"] += 1

        if row["Assigned Player"] == "Unknown":
            counts["unknown"] += 1
        else:
            counts["matched"] += 1

        if row.get("Error"):
            counts["errors"] += 1

    return counts


def _write_reports(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    job_config: Mapping[str, Any],
) -> None:
    reports = output_dir / "Reports"
    reports.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "Original File",
        "Category",
        "Score",
        "Sharpness",
        "Exposure",
        "Contrast",
        "Center Detail",
        "Duplicate",
        "OCR Number",
        "OCR Confidence",
        "OCR Raw",
        "Assigned Player",
        "Assigned Team",
        "Detected Color",
        "Sorted Path",
        "Player Path",
        "Error",
    ]

    with (reports / "diamondvision_report.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metadata_images = []
    for index, row in enumerate(rows, start=1):
        ocr_confidence = float(row.get("OCR Confidence") or 0.0)
        score = float(row.get("Score") or 0.0)
        assigned = row.get("Assigned Player") or "Unknown"
        category = row.get("Category") or ""

        metadata_images.append(
            {
                "id": index,
                "filename": Path(row.get("Sorted Path") or row["Original File"]).name,
                "player": assigned,
                "team": row.get("Assigned Team") or "Unknown",
                "category": category,
                "score": score,
                "ocr": row.get("OCR Number") or "",
                "ocr_confidence": round(ocr_confidence * 100, 1),
                "ocr_raw": row.get("OCR Raw") or "",
                "duplicate": row.get("Duplicate") == "Yes",
                "favorite": False,
                "needs_review": (
                    assigned == "Unknown"
                    or ocr_confidence < 0.50
                    or category == "Reject"
                    or bool(row.get("Error"))
                ),
                "ai_confidence": int(
                    max(0, min(100, round(ocr_confidence * 65 + score * 0.35)))
                ),
                "original_path": row.get("Original File") or "",
                "sorted_path": row.get("Sorted Path") or "",
                "player_path": row.get("Player Path") or "",
                "error": row.get("Error") or "",
            }
        )

    metadata = {
        "worker_schema": "3.0",
        "summary": summary,
        "job_config": {
            key: value
            for key, value in job_config.items()
            if key not in {"dropbox_access_token", "dropbox_refresh_token"}
        },
        "images": metadata_images,
    }

    with (reports / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def process_mobile_job(
    input_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    roster_text: str | Mapping[str, Any] | None = "",
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Process one extracted DiamondVision photo job."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    config = _normalize_job_config(roster_text)

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    team1_name = safe_name(config["team1"])
    team2_name = safe_name(config["team2"])
    mode = config["mode"]
    roster1 = parse_roster_text(str(config.get("roster1", "")))
    roster2 = parse_roster_text(str(config.get("roster2", "")))

    _create_output_folders(output_path, mode, (team1_name, team2_name))

    try:
        detector = DuplicateDetector(
            threshold=int(config.get("duplicate_threshold", 3))
        )
    except TypeError:
        detector = DuplicateDetector()

    images = list(find_images(str(input_path)))
    rows: list[dict[str, Any]] = []
    total = len(images)

    if progress_callback:
        progress_callback(0, total, "Starting DiamondVision Worker 3.0")

    for index, source in enumerate(images, start=1):
        source_path = Path(source)
        basename = source_path.name

        if progress_callback:
            progress_callback(index - 1, total, f"Loading {basename}")

        original_copy = ""
        try:
            original_copy = copy_unique(
                str(source_path),
                str(output_path / "Originals"),
            )

            image = _load_image(str(source_path))
            if image is None:
                raise ValueError("Image decoder returned no image.")

            team_name = team1_name
            active_roster = roster1
            if mode == "two_team" and detect_team_from_color(
                image,
                str(config.get("team1_color", "")),
                str(config.get("team2_color", "")),
            ) == "Team_2":
                team_name = team2_name
                active_roster = roster2

            detected_color = _detect_color_name(image)
            duplicate = _is_duplicate(detector, image, str(source_path))

            if duplicate:
                category = "Duplicates"
                sorted_path = save_display_jpeg(
                    str(source_path),
                    image,
                    str(output_path / "Duplicates"),
                )
                if mode == "two_team":
                    save_display_jpeg(
                        str(source_path),
                        image,
                        str(output_path / "Teams" / team_name / "Duplicates"),
                    )

                row = {
                    "Original File": _relative(original_copy, output_path),
                    "Category": category,
                    "Score": 0.0,
                    "Sharpness": 0.0,
                    "Exposure": 0.0,
                    "Contrast": 0.0,
                    "Center Detail": 0.0,
                    "Duplicate": "Yes",
                    "OCR Number": "",
                    "OCR Confidence": 0.0,
                    "OCR Raw": "",
                    "Assigned Player": "Unknown",
                    "Assigned Team": team_name,
                    "Detected Color": detected_color,
                    "Sorted Path": _relative(sorted_path, output_path),
                    "Player Path": "",
                    "Error": "",
                }
                rows.append(row)

                if progress_callback:
                    progress_callback(index, total, f"Duplicate: {basename}")
                continue

            score, category, components = _score_image(image)
            if category not in {"Best", "Keep", "Reject"}:
                category = "Keep"

            sorted_path = save_display_jpeg(
                str(source_path),
                image,
                str(output_path / category),
            )

            if mode == "two_team":
                save_display_jpeg(
                    str(source_path),
                    image,
                    str(output_path / "Teams" / team_name / category),
                )

            ocr_number = ""
            ocr_confidence = 0.0
            ocr_raw = ""
            assigned_player = "Unknown"
            player_path = ""

            if category in {"Best", "Keep"}:
                ocr_number, ocr_confidence, ocr_raw = read_jersey_number(
                    image,
                    active_roster,
                )
                matched_number = match_roster_number(ocr_number, active_roster)

                if (
                    matched_number
                    and ocr_confidence >= float(config["ocr_min_confidence"])
                ):
                    ocr_number = matched_number
                    assigned_player = active_roster[matched_number]

                player_folder_name = (
                    "Unknown"
                    if assigned_player == "Unknown"
                    else safe_name(f"{ocr_number}_{assigned_player}")
                )
                player_path = save_display_jpeg(
                    str(source_path),
                    image,
                    str(output_path / "Players" / player_folder_name),
                )

                if mode == "two_team":
                    save_display_jpeg(
                        str(source_path),
                        image,
                        str(
                            output_path
                            / "Teams"
                            / team_name
                            / "Players"
                            / player_folder_name
                        ),
                    )

            rows.append(
                {
                    "Original File": _relative(original_copy, output_path),
                    "Category": category,
                    "Score": round(score, 3),
                    "Sharpness": components.get("sharpness", ""),
                    "Exposure": components.get("exposure", ""),
                    "Contrast": components.get("contrast", ""),
                    "Center Detail": components.get("center_detail", ""),
                    "Duplicate": "No",
                    "OCR Number": ocr_number,
                    "OCR Confidence": round(ocr_confidence, 4),
                    "OCR Raw": ocr_raw,
                    "Assigned Player": assigned_player,
                    "Assigned Team": team_name,
                    "Detected Color": detected_color,
                    "Sorted Path": _relative(sorted_path, output_path),
                    "Player Path": _relative(player_path, output_path),
                    "Error": "",
                }
            )

            if progress_callback:
                progress_callback(index, total, f"{category}: {basename}")

        except Exception as error:
            LOGGER.exception("Failed to process %s.", source_path)

            rows.append(
                {
                    "Original File": _relative(original_copy or source_path, output_path),
                    "Category": "Reject",
                    "Score": 0.0,
                    "Sharpness": "",
                    "Exposure": "",
                    "Contrast": "",
                    "Center Detail": "",
                    "Duplicate": "No",
                    "OCR Number": "",
                    "OCR Confidence": 0.0,
                    "OCR Raw": "",
                    "Assigned Player": "Unknown",
                    "Assigned Team": "Unknown",
                    "Detected Color": "unknown",
                    "Sorted Path": "",
                    "Player Path": "",
                    "Error": f"{type(error).__name__}: {error}",
                }
            )

            if progress_callback:
                progress_callback(index, total, f"Error: {basename}")

    eligible = sorted(
        (
            row
            for row in rows
            if row["Category"] in {"Best", "Keep"}
            and row["Duplicate"] == "No"
            and row["Sorted Path"]
        ),
        key=lambda row: float(row["Score"]),
        reverse=True,
    )

    limit = max(0, int(config.get("best_of_tournament_limit", 75)))
    for row in eligible[:limit]:
        try:
            copy_unique(
                str(output_path / row["Sorted Path"]),
                str(output_path / "BestOfTournament"),
            )
        except Exception:
            LOGGER.warning(
                "Could not copy BestOfTournament image %s.",
                row["Sorted Path"],
                exc_info=True,
            )

    summary = _make_summary(rows)
    _write_reports(output_path, rows, summary, config)

    if progress_callback:
        progress_callback(total, total, "Complete")

    LOGGER.info("Processing complete: %s", summary)
    return summary
