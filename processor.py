import os
import cv2
import csv
import shutil
import re
from datetime import datetime

try:
    import rawpy
except ImportError:
    rawpy = None

try:
    import easyocr
except ImportError:
    easyocr = None


SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".nef")
DISPLAY_EXTENSIONS = (".jpg", ".jpeg", ".png")
ocr_reader = None


def safe_name(text):
    text = str(text).strip() or "Unknown"
    for ch in '<>:"/\\|?*':
        text = text.replace(ch, "_")
    return text.replace(" ", "_")


def resize_for_speed(img, max_width=1200):
    h, w = img.shape[:2]
    if w <= max_width:
        return img

    scale = max_width / w
    return cv2.resize(
        img,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA
    )


def load_image_fast(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".nef":
        if rawpy is None:
            return None

        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                half_size=True,
                no_auto_bright=True,
                output_bps=8
            )
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return resize_for_speed(img)

    img = cv2.imread(path)

    if img is None:
        return None

    return resize_for_speed(img)


def save_display_jpeg(original_path, img, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    name = os.path.splitext(os.path.basename(original_path))[0]
    destination = os.path.join(output_folder, f"{name}.jpg")

    if os.path.exists(destination):
        stamp = datetime.now().strftime("%H%M%S%f")
        destination = os.path.join(output_folder, f"{name}_{stamp}.jpg")

    cv2.imwrite(destination, img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

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
    return min((contrast_score(center) * .75) + (contrast_score(img) * .25), 100)


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
        return None

    if ocr_reader is None:
        gpu_enabled = False

        try:
            import torch
            gpu_enabled = torch.cuda.is_available()
        except Exception:
            gpu_enabled = False

        ocr_reader = easyocr.Reader(["en"], gpu=gpu_enabled)

    return ocr_reader


def jersey_crops(img):
    h, w = img.shape[:2]

    crops = [
        ("full_center", img[int(h*.08):int(h*.92), int(w*.12):int(w*.88)]),
        ("torso_center", img[int(h*.18):int(h*.82), int(w*.18):int(w*.82)]),
        ("chest_center", img[int(h*.20):int(h*.68), int(w*.25):int(w*.75)]),
        ("upper_wide", img[int(h*.05):int(h*.72), int(w*.08):int(w*.92)]),
        ("lower_torso", img[int(h*.35):int(h*.92), int(w*.18):int(w*.82)]),
        ("left_body", img[int(h*.12):int(h*.86), int(w*.03):int(w*.58)]),
        ("right_body", img[int(h*.12):int(h*.86), int(w*.42):int(w*.97)]),
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

    upscaled = []
    for name, v in versions:
        h, w = v.shape[:2]
        upscaled.append((
            name,
            cv2.resize(v, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        ))

    return upscaled


def read_jersey_number(img):
    reader = init_ocr()

    if reader is None:
        return "", 0.0, ""

    best_number = ""
    best_conf = 0.0
    raw_hits = []

    try:
        for crop_name, crop in jersey_crops(img):
            for version_name, processed in preprocess_for_ocr(crop):
                found = reader.readtext(
                    processed,
                    detail=1,
                    paragraph=False,
                    allowlist="0123456789"
                )

                for item in found:
                    text = str(item[1])
                    conf = float(item[2])
                    digits = re.sub(r"\D", "", text)

                    if not digits:
                        continue

                    if not (1 <= len(digits) <= 3):
                        continue

                    adjusted_conf = conf

                    # Softball rosters are usually 1–2 digit.
                    if len(digits) <= 2:
                        adjusted_conf += 0.05

                    # Avoid crazy 3-digit false positives unless strong.
                    if len(digits) == 3 and conf < 0.55:
                        continue

                    raw_hits.append(
                        f"{crop_name}/{version_name}:{digits}:{conf:.2f}"
                    )

                    if adjusted_conf > best_conf:
                        best_number = digits
                        best_conf = adjusted_conf

        return best_number, min(best_conf, 1.0), " | ".join(raw_hits)

    except Exception as e:
        return "", 0.0, f"OCR_ERROR:{e}"


def parse_roster_text(text):
    roster = {}

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 2:
            continue

        number = parts[0].replace("#", "")
        name = " ".join(parts[1:])
        roster[number] = name

    return roster


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
        "matched": len([r for r in results if r["Assigned Player"] != "Unknown"])
    }


def detect_team_from_color(img, team1_color="", team2_color=""):
    """
    Simple first-pass jersey color routing.
    Looks at center torso crop and compares dominant HSV color to selected team colors.
    If uncertain, defaults to Team 1.
    """
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



def roster_match_number(ocr_number, roster):
    if not ocr_number:
        return ""

    if ocr_number in roster:
        return ocr_number

    # Common OCR issue: reads 8 when jersey is 88, 1 when jersey is 11, etc.
    doubled = ocr_number + ocr_number
    if doubled in roster:
        return doubled

    # Common OCR issue: drops leading/trailing digit.
    possible = []
    for number in roster.keys():
        if ocr_number in number or number in ocr_number:
            possible.append(number)

    if len(possible) == 1:
        return possible[0]

    return ""

def process_mobile_job(input_dir, output_dir, roster_text="", progress_callback=None):
    """
    Supports both original single-team mode and new two-team mode.

    Original call:
        process_mobile_job(input_dir, output_dir, roster_text)

    New call from app.py:
        process_mobile_job(input_dir, output_dir, job_config)

    job_config:
        {
            "mode": "single" or "two_team",
            "team1": "...",
            "team1_color": "white",
            "roster1": "...",
            "team2": "...",
            "team2_color": "blue",
            "roster2": "..."
        }
    """

    if isinstance(roster_text, dict):
        job_config = roster_text
    else:
        job_config = {
            "mode": "single",
            "team1": "Team",
            "team1_color": "",
            "roster1": roster_text,
            "team2": "Opponent",
            "team2_color": "",
            "roster2": ""
        }

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
        "Reports"
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

    if progress_callback:
        progress_callback(0, total, "Starting DiamondVision...")

    for index, file in enumerate(files, start=1):
        basename = os.path.basename(file)

        if progress_callback:
            progress_callback(index - 1, total, f"Loading {basename}")

        original_path = copy_unique(file, os.path.join(output_dir, "Originals"))

        img = load_image_fast(file)

        if img is None:
            sorted_path = copy_unique(file, os.path.join(output_dir, "Reject"))

            results.append({
                "Original File": file,
                "Category": "Reject",
                "Score": 0,
                "Duplicate": "No",
                "OCR Number": "",
                "OCR Confidence": 0,
                "OCR Raw": "",
                "Assigned Player": "Unknown",
                "Assigned Team": "Unknown",
                "Sorted Path": sorted_path,
                "Player Path": ""
            })

            continue

        assigned_team_key = "Team_1"
        assigned_team_name = team1_name
        active_roster = roster1

        if mode == "two_team":
            assigned_team_key = detect_team_from_color(img, team1_color, team2_color)

            if assigned_team_key == "Team_2":
                assigned_team_name = team2_name
                active_roster = roster2
            else:
                assigned_team_name = team1_name
                active_roster = roster1

        img_hash = average_hash(img)
        duplicate = any(hamming_distance(img_hash, h) <= 3 for h in hashes)

        if duplicate:
            display_path = save_display_jpeg(file, img, os.path.join(output_dir, "Duplicates"))

            results.append({
                "Original File": original_path,
                "Category": "Duplicates",
                "Score": 0,
                "Duplicate": "Yes",
                "OCR Number": "",
                "OCR Confidence": 0,
                "OCR Raw": "",
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

        display_path = save_display_jpeg(file, img, os.path.join(output_dir, category))

        if mode == "two_team":
            save_display_jpeg(
                file,
                img,
                os.path.join(output_dir, "Teams", assigned_team_name, category)
            )

        ocr_number = ""
        ocr_conf = 0
        ocr_raw = ""
        assigned = "Unknown"
        player_path = ""

        if category in ("Best", "Keep"):
            ocr_number, ocr_conf, ocr_raw = read_jersey_number(img)

            matched_number = roster_match_number(ocr_number, active_roster)

            if matched_number and ocr_conf >= .25:
                ocr_number = matched_number
                assigned = active_roster[matched_number]

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
            "Assigned Player": assigned,
            "Assigned Team": assigned_team_name,
            "Sorted Path": display_path,
            "Player Path": player_path
        })

        if progress_callback:
            progress_callback(index, total, f"{category}: {basename}")

    best = sorted(
        [
            r for r in results
            if r["Category"] in ("Best", "Keep") and r["Duplicate"] == "No"
        ],
        key=lambda x: x["Score"],
        reverse=True
    )

    for r in best[:75]:
        try:
            copy_unique(r["Sorted Path"], os.path.join(output_dir, "BestOfTournament"))
        except Exception:
            pass

    report = os.path.join(output_dir, "Reports", "diamondvision_report.csv")

    with open(report, "w", newline="") as f:
        fieldnames = [
            "Original File",
            "Category",
            "Score",
            "Duplicate",
            "OCR Number",
            "OCR Confidence",
            "OCR Raw",
            "Assigned Player",
            "Assigned Team",
            "Sorted Path",
            "Player Path"
        ]

        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    summary = summarize_results(results)

    if progress_callback:
        progress_callback(total, total, "Complete")

    return summary

def unzip_file(zip_path, out_dir):
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


def zip_folder(folder, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(folder):
            for file in files:
                full = os.path.join(root, file)
                rel = os.path.relpath(full, folder)
                z.write(full, rel)


def list_files(folder):
    found = []

    for root, _, files in os.walk(folder):
        for file in files:
            full = os.path.join(root, file)
            rel = os.path.relpath(full, folder).replace("\\", "/")
            found.append(rel)

    return sorted(found)


def count_extensions(files):
    counts = {}

    for file in files:
        ext = os.path.splitext(file)[1].lower() or "no_ext"
        counts[ext] = counts.get(ext, 0) + 1

    return counts


def dropbox_download(dbx, dropbox_path, local_path):
    log(f"Downloading from Dropbox: {dropbox_path}")

    _, res = dbx.files_download(dropbox_path)

    with open(local_path, "wb") as f:
        f.write(res.content)

    log(f"Downloaded to: {local_path}")


def dropbox_upload(dbx, local_path, dropbox_path):
    log(f"Uploading to Dropbox: {dropbox_path}")

    with open(local_path, "rb") as f:
        dbx.files_upload(
            f.read(),
            dropbox_path,
            mode=dropbox.files.WriteMode.overwrite
        )

    log("Dropbox upload complete.")


def get_rawpy_status():
    try:
        import rawpy

        return {
            "rawpy_available": True,
            "rawpy_version": getattr(rawpy, "__version__", "unknown"),
            "rawpy_error": ""
        }

    except Exception as e:
        return {
            "rawpy_available": False,
            "rawpy_version": "",
            "rawpy_error": str(e)
        }


def handler(job):
    log("=" * 60)
    log("Handler started.")
    log("=" * 60)

    rawpy_status = get_rawpy_status()

    log(f"rawpy_available={rawpy_status['rawpy_available']}")
    log(f"rawpy_version={rawpy_status['rawpy_version']}")

    if rawpy_status["rawpy_error"]:
        log(f"rawpy_error={rawpy_status['rawpy_error']}")

    data = job.get("input", {}) or {}

    input_zip_dropbox_path = data.get("input_zip_dropbox_path")
    output_zip_dropbox_path = data.get("output_zip_dropbox_path")
    job_config = data.get("job_config", {})

    dbx = get_dropbox_client(data)

    if dbx is None:
        return {
            "error": "Missing Dropbox credentials. Provide DROPBOX_APP_KEY, DROPBOX_APP_SECRET, and DROPBOX_REFRESH_TOKEN, or provide dropbox_access_token."
        }

    if not input_zip_dropbox_path:
        return {"error": "Missing input_zip_dropbox_path"}

    if not output_zip_dropbox_path:
        return {"error": "Missing output_zip_dropbox_path"}

    with tempfile.TemporaryDirectory() as tmp:
        input_zip = os.path.join(tmp, "input.zip")
        output_zip = os.path.join(tmp, "results.zip")
        input_dir = os.path.join(tmp, "input")
        output_dir = os.path.join(tmp, "output")

        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        dropbox_download(dbx, input_zip_dropbox_path, input_zip)
        unzip_file(input_zip, input_dir)

        input_files = list_files(input_dir)
        input_counts = count_extensions(input_files)

        log(f"Input files found: {len(input_files)}")
        log(f"Input extension counts: {input_counts}")

        nef_files = [f for f in input_files if f.lower().endswith(".nef")]
        log(f"NEF files found: {len(nef_files)}")

        for nef in nef_files[:20]:
            log(f"NEF input: {nef}")

        log("Starting processor.")

        summary = process_mobile_job(
            input_dir,
            output_dir,
            job_config,
            progress_callback=None
        )

        log(f"Processor complete. Summary: {summary}")

        output_files = list_files(output_dir)
        output_counts = count_extensions(output_files)

        log(f"Output files created: {len(output_files)}")
        log(f"Output extension counts: {output_counts}")

        jpg_outputs = [
            f for f in output_files
            if f.lower().endswith((".jpg", ".jpeg"))
        ]

        log(f"Display JPG outputs: {len(jpg_outputs)}")

        zip_folder(output_dir, output_zip)
        dropbox_upload(dbx, output_zip, output_zip_dropbox_path)

        log("Handler finished.")

        return {
            "worker_version": WORKER_VERSION,
            "summary": summary,
            "output_zip_dropbox_path": output_zip_dropbox_path,
            "rawpy": rawpy_status,
            "input_file_count": len(input_files),
            "input_extension_counts": input_counts,
            "nef_files_found": len(nef_files),
            "nef_files_sample": nef_files[:20],
            "output_file_count": len(output_files),
            "output_extension_counts": output_counts,
            "jpg_output_count": len(jpg_outputs)
        }


runpod.serverless.start({"handler": handler})
