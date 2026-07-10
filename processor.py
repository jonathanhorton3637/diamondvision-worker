import os
import zipfile
import tempfile
import runpod
import dropbox

from processor import process_mobile_job


WORKER_VERSION = "DiamondVision Worker v2.1 Refresh Token Dropbox"


def log(message):
    print(f"[{WORKER_VERSION}] {message}", flush=True)


def get_dropbox_client(data):
    dropbox_access_token = data.get("dropbox_access_token") or os.environ.get("DROPBOX_ACCESS_TOKEN", "")

    dropbox_app_key = data.get("dropbox_app_key") or os.environ.get("DROPBOX_APP_KEY", "")
    dropbox_app_secret = data.get("dropbox_app_secret") or os.environ.get("DROPBOX_APP_SECRET", "")
    dropbox_refresh_token = data.get("dropbox_refresh_token") or os.environ.get("DROPBOX_REFRESH_TOKEN", "")

    if dropbox_refresh_token and dropbox_app_key and dropbox_app_secret:
        log("Using Dropbox refresh token authentication.")
        return dropbox.Dropbox(
            oauth2_refresh_token=dropbox_refresh_token.strip(),
            app_key=dropbox_app_key.strip(),
            app_secret=dropbox_app_secret.strip()
        )

    if dropbox_access_token:
        log("Using Dropbox access token authentication.")
        return dropbox.Dropbox(dropbox_access_token.strip())

    return None


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
