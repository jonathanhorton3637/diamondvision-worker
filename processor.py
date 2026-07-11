import os
import json
import cv2
import csv
import shutil
import re
from datetime import datetime


PROCESSOR_VERSION = "DiamondVision Processor v2.1 OCR"


def log(message):
    print(f"[{PROCESSOR_VERSION}] {message}", flush=True)


try:
    import rawpy
    RAWPY_AVAILABLE = True
    RAWPY_IMPORT_ERROR = ""
    RAWPY_VERSION = getattr(rawpy, "__version__", "unknown")
    log("rawpy successfully imported.")
    log(f"rawpy version: {RAWPY_VERSION}")
except Exception as e:
    rawpy = None
    RAWPY_AVAILABLE = False
    RAWPY_IMPORT_ERROR = str(e)
    RAWPY_VERSION = ""
    log("rawpy NOT available.")
    log(f"rawpy import error: {RAWPY_IMPORT_ERROR}")


try:
    import easyocr
except ImportError:
    easyocr = None


SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".nef")
RAW_EXTENSIONS = (".nef",)
DISPLAY_EXTENSIONS = (".jpg", ".jpeg", ".png")
ocr_reader = None


def safe_name(text):
    text = str(text).strip() or "Unknown"
    for ch in '<>:"/\\|?*#':
        text = text.replace(ch, "_")
    return text.replace(" ", "_")


def resize_for_speed(img, max_width=1400):
    h, w = img.shape[:2]
    if w <= max_width:
        return img

    scale = max_width / w
    return cv2.resize(
        img,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA
    )


def load_image_fast(path, raw_debug=None):
    basename = os.path.basename(path)
    ext = os.path.splitext(path)[1].lower()

    log(f"Loading image: {basename}")
    log(f"Extension detected: {ext}")

    if ext in RAW_EXTENSIONS:
        if raw_debug is not None:
            raw_debug["raw_seen"] += 1

        if rawpy is None:
            reason = f"rawpy unavailable: {RAWPY_IMPORT_ERROR}"
            log(reason)

            if raw_debug is not None:
                raw_debug["raw_failed"] += 1
                raw_debug["raw_failures"].append({
                    "file": basename,
                    "reason": reason
                })

            return None

        try:
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    half_size=True,
                    no_auto_bright=True,
                    output_bps=8
                )

            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            img = resize_for_speed(img)

            log(f"RAW/NEF decoded successfully: {basename}")

            if raw_debug is not None:
                raw_debug["raw_decoded"] += 1

            return img

        except Exception as e:
            log(f"RAW/NEF decode failed for {basename}: {e}")

            if raw_debug is not None:
                raw_debug["raw_failed"] += 1
                raw_debug["raw_failures"].append({
                    "file": basename,
                    "reason": str(e)
                })

            return None

    img = cv2.imread(path)

    if img is None:
        log(f"cv2 failed to load image: {basename}")
        return None

    return resize_for_speed(img)


def save_display_jpeg(original_path, img, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    name = os.path.splitext(os.path.basename(original_path))[0]
    destination = os.path.join(output_folder, f"{name}.jpg")

    if os.path.exists(destination):
        stamp = datetime.now().strftime("%H%M%S%f")
        destination = os.path.join(output_folder, f"{name}_{stamp}.jpg")

    ok = cv2.imwrite(destination, img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    if not ok:
        raise RuntimeError(f"Failed to write JPEG: {destination}")

    return destination


def find_images(folder):
    files = []

    for root, _, names in os.walk(folder):
        for name in names:
            if name.lower().endswith(SUPPORTED_EXTENSIONS):
                files.append(os.path.join(root, name))

    return sorted(files)


def sharpness_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return min(cv2.Laplacian(gray, cv2.CV_64F).var() / 8, 100)


def exposure_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = gray.mean()

    if 80 <= mean <= 180:
        return 100

    if mean < 80:
        return max(0, mean / 80 * 100)

    return max(0, (255 - mean) / 75 * 100)


def contrast_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return min(gray.std() * 2, 100)


def center_detail_score(img):
    h, w = img.shape[:2]
    center = img[int(h * .18):int(h * .82), int(w * .18):int(w * .82)]

    return min(
        (contrast_score(center) * .75) + (contrast_score(img) * .25),
        100
    )


def total_score(img):
    return max(0, min(
        sharpness_score(img) * .50 +
        exposure_score(img) * .15 +
        contrast_score(img) * .15 +
        center_detail_score(img) * .20,
        100
    ))


def classify(score):
    if score >= 70:
        return "Best"

    if score >= 38:
        return "Keep"

    return "Reject"


def average_hash(img, hash_size=8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    return (small > small.mean()).flatten()


def hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))


def init_ocr():
    global ocr_reader

    if easyocr is None:
        log("easyocr is not available.")
        return None

    if ocr_reader is None:
        gpu_enabled = False

        try:
            import torch
            gpu_enabled = torch.cuda.is_available()
        except Exception:
            gpu_enabled = False

        log(f"Initializing EasyOCR. GPU={gpu_enabled}")
        ocr_reader = easyocr.Reader(["en"], gpu=gpu_enabled)

    return ocr_reader


def jersey_crops(img):
    h, w = img.shape[:2]

    crops = [
        ("full", img),
        ("full_center", img[int(h*.05):int(h*.95), int(w*.08):int(w*.92)]),
        ("torso_center", img[int(h*.15):int(h*.88), int(w*.15):int(w*.85)]),
        ("chest_center", img[int(h*.18):int(h*.70), int(w*.22):int(w*.78)]),
        ("upper_wide", img[int(h*.05):int(h*.76), int(w*.05):int(w*.95)]),
        ("middle_wide", img[int(h*.20):int(h*.82), int(w*.05):int(w*.95)]),
        ("lower_torso", img[int(h*.32):int(h*.95), int(w*.15):int(w*.85)]),
        ("left_body", img[int(h*.08):int(h*.90), int(w*.00):int(w*.62)]),
        ("right_body", img[int(h*.08):int(h*.90), int(w*.38):int(w*1.00)]),
        ("center_tall", img[int(h*.00):int(h*1.00), int(w*.25):int(w*.75)]),
        ("center_square", img[int(h*.18):int(h*.82), int(w*.20):int(w*.80)]),
    ]

    good = []

    for name, crop in crops:
        if crop is None:
            continue

        ch, cw = crop.shape[:2]

        if ch >= 40 and cw >= 40:
            good.append((name, crop))

    return good


def preprocess_for_ocr(crop):
    versions = []

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    versions.append(("gray", gray))

    eq = cv2.equalizeHist(gray)
    versions.append(("equalized", eq))

    blur = cv2.GaussianBlur(eq, (3, 3), 0)

    adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8
    )

    versions.append(("adaptive", adaptive))
    versions.append(("adaptive_inv", cv2.bitwise_not(adaptive)))

    _, otsu = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    versions.append(("otsu", otsu))
    versions.append(("otsu_inv", cv2.bitwise_not(otsu)))

    high_contrast = cv2.convertScaleAbs(gray, alpha=1.8, beta=10)
    versions.append(("high_contrast", high_contrast))

    upscaled = []

    for name, v in versions:
        h, w = v.shape[:2]

        if h < 900 and w < 900:
            upscaled.append((
                name + "_2x",
                cv2.resize(v, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            ))

        upscaled.append((name, v))

    return upscaled


def clean_ocr_digits(text):
    text = str(text or "").upper()

    replacements = {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "|": "1",
        "S": "5",
        "B": "8",
        "G": "6",
        "Z": "2",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return re.sub(r"\D", "", text)


def read_jersey_number(img):
    reader = init_ocr()

    if reader is None:
        return "", 0.0, "OCR_DISABLED"

    candidates = []
    raw_hits = []

    try:
        for crop_name, crop in jersey_crops(img):
            for version_name, processed in preprocess_for_ocr(crop):
                found = reader.readtext(
                    processed,
                    detail=1,
                    paragraph=False,
                    allowlist="0123456789#OQDISBLZG|"
                )

                for item in found:
                    text = str(item[1])
                    conf = float(item[2])
                    digits = clean_ocr_digits(text)

                    if not digits:
                        continue

                    if not (1 <= len(digits) <= 3):
                        continue

                    adjusted_conf = conf

                    if len(digits) <= 2:
                        adjusted_conf += 0.08

                    if len(digits) == 3 and conf < 0.45:
                        continue

                    raw_hits.append(
                        f"{crop_name}/{version_name}:{text}->{digits}:{conf:.2f}"
                    )

                    candidates.append({
                        "digits": digits,
                        "confidence": min(adjusted_conf, 1.0),
                        "raw_confidence": conf,
                        "source": f"{crop_name}/{version_name}",
                        "text": text
                    })

        if not candidates:
            return "", 0.0, ""

        candidates = sorted(
            candidates,
            key=lambda x: x["confidence"],
            reverse=True
        )

        best = candidates[0]

        return (
            best["digits"],
            min(best["confidence"], 1.0),
            " | ".join(raw_hits)
        )

    except Exception as e:
        return "", 0.0, f"OCR_ERROR:{e}"


def parse_roster_text(text):
    """
    Accepts:
    12 John Smith
    John Smith 12
    #12 John Smith
    John Smith #12
    12, John Smith
    John Smith, 12
    12 - John Smith
    John Smith - 12
    12: John Smith
    John Smith: 12
    """

    roster = {}

    for line in str(text or "").splitlines():
        original = line.strip()

        if not original:
            continue

        line = original.replace("\t", " ")
        line = re.sub(r"\s+", " ", line).strip()

        number = ""
        name = ""

        parts = [
            p.strip()
            for p in re.split(r"\s*[,:\-–—|]\s*", line)
            if p.strip()
        ]

        if len(parts) >= 2:
            first = parts[0]
            last = parts[-1]

            first_num = re.fullmatch(r"#?\d{1,3}", first)
            last_num = re.fullmatch(r"#?\d{1,3}", last)

            if first_num:
                number = first.replace("#", "")
                name = " ".join(parts[1:]).strip()

            elif last_num:
                number = last.replace("#", "")
                name = " ".join(parts[:-1]).strip()

        if not number:
            m = re.match(r"^#?(\d{1,3})\s+(.+)$", line)
            if m:
                number = m.group(1)
                name = m.group(2).strip()

        if not number:
            m = re.match(r"^(.+?)\s+#?(\d{1,3})$", line)
            if m:
                name = m.group(1).strip()
                number = m.group(2)

        if not number or not name:
            log(f"Roster line skipped: {original}")
            continue

        name = re.sub(r"\s+", " ", name).strip(" ,:-–—|")
        number = number.strip().replace("#", "")

        if not number.isdigit():
            log(f"Roster line skipped, invalid number: {original}")
            continue

        roster[number] = name

    log(f"Roster parsed: {len(roster)} players")
    return roster


def roster_match_number(ocr_number, roster, ocr_conf=0.0, ocr_raw=""):
    if not roster:
        return "", "no_roster"

    ocr_number = clean_ocr_digits(ocr_number)
    raw_digits = clean_ocr_digits(ocr_raw)

    if not ocr_number and not raw_digits:
        return "", "no_ocr_number"

    if ocr_number in roster:
        return ocr_number, "exact"

    if raw_digits:
        for number in roster.keys():
            if number in raw_digits:
                return number, "raw_contains_roster_number"

    if ocr_number:
        doubled = ocr_number + ocr_number
        if doubled in roster:
            return doubled, "doubled_digit"

    possible = []

    for number in roster.keys():
        if ocr_number and (ocr_number in number or number in ocr_number):
            possible.append(number)

    if len(possible) == 1:
        return possible[0], "partial_unique"

    if len(possible) > 1:
        possible_sorted = sorted(possible, key=lambda x: abs(len(x) - len(ocr_number)))
        return possible_sorted[0], "partial_best_guess"

    if len(ocr_number) == 1:
        ending = [n for n in roster.keys() if n.endswith(ocr_number)]
        if len(ending) == 1:
            return ending[0], "single_digit_suffix_unique"

    return "", "no_match"


def copy_unique(src, folder):
    os.makedirs(folder, exist_ok=True)

    dest = os.path.join(folder, os.path.basename(src))

    if os.path.exists(dest):
        name, ext = os.path.splitext(os.path.basename(src))
        stamp = datetime.now().strftime("%H%M%S%f")
        dest = os.path.join(folder, f"{name}_{stamp}{ext}")

    shutil.copy2(src, dest)

    return dest


def summarize_results(results):
    return {
        "total": len(results),
        "best": len([r for r in results if r["Category"] == "Best"]),
        "keep": len([r for r in results if r["Category"] == "Keep"]),
        "reject": len([r for r in results if r["Category"] == "Reject"]),
        "duplicates": len([r for r in results if r["Category"] == "Duplicates"]),
        "matched": len([r for r in results if r["Assigned Player"] != "Unknown"]),
    }


def detect_team_from_color(img, team1_color="", team2_color=""):
    if not team1_color and not team2_color:
        return "Team_1"

    h, w = img.shape[:2]
    crop = img[int(h*.20):int(h*.80), int(w*.25):int(w*.75)]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    mean_h = hsv[:, :, 0].mean()
    mean_s = hsv[:, :, 1].mean()
    mean_v = hsv[:, :, 2].mean()

    detected = "unknown"

    if mean_v > 185 and mean_s < 70:
        detected = "white"
    elif mean_v < 80:
        detected = "black"
    elif mean_s < 55 and 80 <= mean_v <= 185:
        detected = "gray"
    elif mean_h < 10 or mean_h > 165:
        detected = "red"
    elif 90 <= mean_h <= 130:
        detected = "blue"
    elif 35 <= mean_h <= 85:
        detected = "green"
    elif 18 <= mean_h <= 35:
        detected = "yellow"

    if team1_color and detected == team1_color:
        return "Team_1"

    if team2_color and detected == team2_color:
        return "Team_2"

    return "Team_1"


def build_job_config(roster_text):
    if isinstance(roster_text, dict):
        return roster_text

    return {
        "mode": "single",
        "team1": "Team",
        "team1_color": "",
        "roster1": roster_text,
        "team2": "Opponent",
        "team2_color": "",
        "roster2": ""
    }


def process_mobile_job(input_dir, output_dir, roster_text="", progress_callback=None):
    log("=" * 60)
    log("Starting process_mobile_job.")
    log(f"rawpy_available={RAWPY_AVAILABLE}")
    log(f"rawpy_version={RAWPY_VERSION}")

    raw_debug = {
        "rawpy_available": RAWPY_AVAILABLE,
        "rawpy_version": RAWPY_VERSION,
        "rawpy_import_error": RAWPY_IMPORT_ERROR,
        "raw_seen": 0,
        "raw_decoded": 0,
        "raw_failed": 0,
        "raw_failures": [],
    }

    job_config = build_job_config(roster_text)

    mode = job_config.get("mode", "single")

    team1_name = safe_name(job_config.get("team1", "Team_1"))
    team2_name = safe_name(job_config.get("team2", "Team_2"))

    team1_color = job_config.get("team1_color", "")
    team2_color = job_config.get("team2_color", "")

    roster1 = parse_roster_text(job_config.get("roster1", ""))
    roster2 = parse_roster_text(job_config.get("roster2", ""))

    hashes = []
    results = []

    base_folders = [
        "Originals",
        "Best",
        "Keep",
        "Reject",
        "Duplicates",
        "BestOfTournament",
        "Players",
        "Players/Unknown",
        "Favorites",
        "Reports",
    ]

    for folder in base_folders:
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    if mode == "two_team":
        team_folders = [
            f"Teams/{team1_name}/Players/Unknown",
            f"Teams/{team1_name}/Best",
            f"Teams/{team1_name}/Keep",
            f"Teams/{team1_name}/Reject",
            f"Teams/{team2_name}/Players/Unknown",
            f"Teams/{team2_name}/Best",
            f"Teams/{team2_name}/Keep",
            f"Teams/{team2_name}/Reject",
        ]

        for folder in team_folders:
            os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    files = find_images(input_dir)
    total = len(files)

    log(f"Found {total} supported image(s).")

    if progress_callback:
        progress_callback(0, total, "Starting DiamondVision...")

    for index, file in enumerate(files, start=1):
        basename = os.path.basename(file)

        log("-" * 60)
        log(f"Processing {index}/{total}: {basename}")

        if progress_callback:
            progress_callback(index - 1, total, f"Loading {basename}")

        original_path = copy_unique(file, os.path.join(output_dir, "Originals"))
        img = load_image_fast(file, raw_debug=raw_debug)

        if img is None:
            results.append({
                "Original File": original_path,
                "Category": "Reject",
                "Score": 0,
                "Duplicate": "No",
                "OCR Number": "",
                "OCR Confidence": 0,
                "OCR Raw": "",
                "OCR Match Method": "decode_failed",
                "Assigned Player": "Unknown",
                "Assigned Team": "Unknown",
                "Sorted Path": "",
                "Player Path": ""
            })

            if progress_callback:
                progress_callback(index, total, f"Rejected unreadable file: {basename}")

            continue

        assigned_team_name = team1_name
        active_roster = roster1

        if mode == "two_team":
            assigned_team_key = detect_team_from_color(img, team1_color, team2_color)

            if assigned_team_key == "Team_2":
                assigned_team_name = team2_name
                active_roster = roster2

        img_hash = average_hash(img)
        duplicate = any(hamming_distance(img_hash, h) <= 3 for h in hashes)

        if duplicate:
            display_path = save_display_jpeg(
                file,
                img,
                os.path.join(output_dir, "Duplicates")
            )

            results.append({
                "Original File": original_path,
                "Category": "Duplicates",
                "Score": 0,
                "Duplicate": "Yes",
                "OCR Number": "",
                "OCR Confidence": 0,
                "OCR Raw": "",
                "OCR Match Method": "duplicate_skipped",
                "Assigned Player": "Unknown",
                "Assigned Team": assigned_team_name,
                "Sorted Path": display_path,
                "Player Path": ""
            })

            if progress_callback:
                progress_callback(index, total, f"Duplicate: {basename}")

            continue

        hashes.append(img_hash)

        score = total_score(img)
        category = classify(score)

        display_path = save_display_jpeg(
            file,
            img,
            os.path.join(output_dir, category)
        )

        if mode == "two_team":
            save_display_jpeg(
                file,
                img,
                os.path.join(output_dir, "Teams", assigned_team_name, category)
            )

        ocr_number = ""
        ocr_conf = 0
        ocr_raw = ""
        match_method = ""
        assigned = "Unknown"
        player_path = ""

        ocr_number, ocr_conf, ocr_raw = read_jersey_number(img)
        matched_number, match_method = roster_match_number(
            ocr_number,
            active_roster,
            ocr_conf=ocr_conf,
            ocr_raw=ocr_raw
        )

        if matched_number:
            ocr_number = matched_number
            assigned = active_roster[matched_number]

        if category in ("Best", "Keep") or assigned != "Unknown":
            if assigned == "Unknown":
                player_folder = os.path.join(output_dir, "Players", "Unknown")
            else:
                player_folder = os.path.join(
                    output_dir,
                    "Players",
                    safe_name(f"{ocr_number}_{assigned}")
                )

            player_path = save_display_jpeg(file, img, player_folder)

            if mode == "two_team":
                if assigned == "Unknown":
                    team_player_folder = os.path.join(
                        output_dir,
                        "Teams",
                        assigned_team_name,
                        "Players",
                        "Unknown"
                    )
                else:
                    team_player_folder = os.path.join(
                        output_dir,
                        "Teams",
                        assigned_team_name,
                        "Players",
                        safe_name(f"{ocr_number}_{assigned}")
                    )

                save_display_jpeg(file, img, team_player_folder)

        results.append({
            "Original File": original_path,
            "Category": category,
            "Score": round(score, 2),
            "Duplicate": "No",
            "OCR Number": ocr_number,
            "OCR Confidence": round(ocr_conf, 3),
            "OCR Raw": ocr_raw,
            "OCR Match Method": match_method,
            "Assigned Player": assigned,
            "Assigned Team": assigned_team_name,
            "Sorted Path": display_path,
            "Player Path": player_path
        })

        log(
            f"Finished {basename}: category={category}, score={round(score, 2)}, "
            f"ocr={ocr_number}, conf={round(ocr_conf, 3)}, "
            f"match={match_method}, player={assigned}"
        )

        if progress_callback:
            progress_callback(index, total, f"{category}: {basename}")

    best = sorted(
        [
            r for r in results
            if r["Category"] in ("Best", "Keep")
            and r["Duplicate"] == "No"
            and r.get("Sorted Path")
        ],
        key=lambda x: x["Score"],
        reverse=True
    )

    for r in best[:75]:
        try:
            copy_unique(r["Sorted Path"], os.path.join(output_dir, "BestOfTournament"))
        except Exception as e:
            log(f"BestOfTournament copy failed: {e}")

    report = os.path.join(output_dir, "Reports", "diamondvision_report.csv")

    with open(report, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "Original File",
            "Category",
            "Score",
            "Duplicate",
            "OCR Number",
            "OCR Confidence",
            "OCR Raw",
            "OCR Match Method",
            "Assigned Player",
            "Assigned Team",
            "Sorted Path",
            "Player Path"
        ]

        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    metadata_images = []

    for r in results:
        filename = os.path.basename(r.get("Sorted Path") or r.get("Original File") or "")

        ocr_conf = float(r.get("OCR Confidence") or 0)
        score = float(r.get("Score") or 0)
        duplicate = r.get("Duplicate") == "Yes"
        category = r.get("Category", "")
        assigned_player = r.get("Assigned Player", "Unknown") or "Unknown"

        needs_review = (
            assigned_player == "Unknown"
            or ocr_conf < 0.35
            or category == "Reject"
        )

        ai_confidence = int(max(
            0,
            min(100, round(((ocr_conf * 100) * 0.65) + (score * 0.35)))
        ))

        metadata_images.append({
            "id": os.path.splitext(filename)[0],
            "filename": filename,
            "player": assigned_player,
            "team": r.get("Assigned Team", "Unknown"),
            "category": category,
            "score": score,
            "ocr": r.get("OCR Number", ""),
            "ocr_confidence": round(ocr_conf * 100, 1),
            "ocr_raw": r.get("OCR Raw", ""),
            "ocr_match_method": r.get("OCR Match Method", ""),
            "duplicate": duplicate,
            "favorite": False,
            "needs_review": needs_review,
            "ai_confidence": ai_confidence,
            "original_path": r.get("Original File", ""),
            "sorted_path": r.get("Sorted Path", ""),
            "player_path": r.get("Player Path", "")
        })

    summary = summarize_results(results)

    metadata = {
        "version": "4.1",
        "processor_version": PROCESSOR_VERSION,
        "summary": summary,
        "raw_support": raw_debug,
        "images": metadata_images
    }

    metadata_path = os.path.join(output_dir, "Reports", "metadata.json")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if progress_callback:
        progress_callback(total, total, "Complete")

    log("=" * 60)
    log("process_mobile_job complete.")
    log(f"Summary: {summary}")
    log(f"RAW support: {raw_debug}")
    log("=" * 60)

    return summary