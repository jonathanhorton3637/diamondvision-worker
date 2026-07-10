import os
import tempfile
import zipfile

import dropbox
import runpod

from processor import process_mobile_job


WORKER_VERSION = "DiamondVision Worker v2.1 Refresh Token Dropbox"


def log(message):
    print(f"[{WORKER_VERSION}] {message}", flush=True)


def get_dropbox_client(data):
    access_token = (
        data.get("dropbox_access_token")
        or os.environ.get("DROPBOX_ACCESS_TOKEN", "")
    )

    app_key = (
        data.get("dropbox_app_key")
        or os.environ.get("DROPBOX_APP_KEY", "")
    )

    app_secret = (
        data.get("dropbox_app_secret")
        or os.environ.get("DROPBOX_APP_SECRET", "")
    )

    refresh_token = (
        data.get("dropbox_refresh_token")
        or os.environ.get("DROPBOX_REFRESH_TOKEN", "")
    )

    if refresh_token and app_key and app_secret:
        log("Using Dropbox refresh token authentication.")

        return dropbox.Dropbox(
            oauth2_refresh_token=refresh_token.strip(),
            app_key=app_key.strip(),
            app_secret=app_secret.strip()
        )

    if access_token:
        log("Using Dropbox access token authentication.")

        return dropbox.Dropbox(
            access_token.strip()
        )

    return None


def unzip_file(zip_path, output_directory):
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_directory)


def zip_folder(folder_path, zip_path):
    with zipfile.ZipFile(
        zip_path,
        "w",
        zipfile.ZIP_DEFLATED
    ) as archive:
        for root, _, files in os.walk(folder_path):
            for filename in files:
                full_path = os.path.join(root, filename)
                relative_path = os.path.relpath(
                    full_path,
                    folder_path
                )

                archive.write(
                    full_path,
                    relative_path
                )


def list_files(folder_path):
    found_files = []

    for root, _, files in os.walk(folder_path):
        for filename in files:
            full_path = os.path.join(root, filename)

            relative_path = os.path.relpath(
                full_path,
                folder_path
            ).replace("\\", "/")

            found_files.append(relative_path)

    return sorted(found_files)


def count_extensions(files):
    counts = {}

    for filename in files:
        extension = (
            os.path.splitext(filename)[1].lower()
            or "no_ext"
        )

        counts[extension] = counts.get(extension, 0) + 1

    return counts


def dropbox_download(
    dbx,
    dropbox_path,
    local_path
):
    log(f"Downloading from Dropbox: {dropbox_path}")

    _, response = dbx.files_download(
        dropbox_path
    )

    with open(local_path, "wb") as file:
        file.write(response.content)

    log(f"Downloaded to: {local_path}")


def upload_small_file(
    dbx,
    local_path,
    dropbox_path
):
    with open(local_path, "rb") as file:
        dbx.files_upload(
            file.read(),
            dropbox_path,
            mode=dropbox.files.WriteMode.overwrite
        )


def upload_large_file(
    dbx,
    local_path,
    dropbox_path,
    chunk_size=8 * 1024 * 1024
):
    file_size = os.path.getsize(local_path)

    with open(local_path, "rb") as file:
        first_chunk = file.read(chunk_size)

        session = dbx.files_upload_session_start(
            first_chunk
        )

        cursor = dropbox.files.UploadSessionCursor(
            session_id=session.session_id,
            offset=file.tell()
        )

        commit = dropbox.files.CommitInfo(
            path=dropbox_path,
            mode=dropbox.files.WriteMode.overwrite
        )

        while file.tell() < file_size:
            remaining = file_size - file.tell()
            chunk = file.read(
                min(chunk_size, remaining)
            )

            if file.tell() >= file_size:
                dbx.files_upload_session_finish(
                    chunk,
                    cursor,
                    commit
                )
            else:
                dbx.files_upload_session_append_v2(
                    chunk,
                    cursor
                )

                cursor.offset = file.tell()


def dropbox_upload(
    dbx,
    local_path,
    dropbox_path
):
    log(f"Uploading to Dropbox: {dropbox_path}")

    file_size = os.path.getsize(local_path)

    if file_size <= 140 * 1024 * 1024:
        upload_small_file(
            dbx,
            local_path,
            dropbox_path
        )
    else:
        upload_large_file(
            dbx,
            local_path,
            dropbox_path
        )

    log("Dropbox upload complete.")


def get_rawpy_status():
    try:
        import rawpy

        return {
            "rawpy_available": True,
            "rawpy_version": getattr(
                rawpy,
                "__version__",
                "unknown"
            ),
            "rawpy_error": ""
        }

    except Exception as error:
        return {
            "rawpy_available": False,
            "rawpy_version": "",
            "rawpy_error": str(error)
        }


def handler(job):
    log("=" * 60)
    log("Handler started.")
    log("=" * 60)

    rawpy_status = get_rawpy_status()

    log(
        "rawpy_available="
        f"{rawpy_status['rawpy_available']}"
    )

    log(
        "rawpy_version="
        f"{rawpy_status['rawpy_version']}"
    )

    if rawpy_status["rawpy_error"]:
        log(
            "rawpy_error="
            f"{rawpy_status['rawpy_error']}"
        )

    data = job.get("input", {}) or {}

    input_zip_dropbox_path = data.get(
        "input_zip_dropbox_path"
    )

    output_zip_dropbox_path = data.get(
        "output_zip_dropbox_path"
    )

    job_config = data.get(
        "job_config",
        {}
    )

    if not input_zip_dropbox_path:
        return {
            "error": "Missing input_zip_dropbox_path",
            "worker_version": WORKER_VERSION
        }

    if not output_zip_dropbox_path:
        return {
            "error": "Missing output_zip_dropbox_path",
            "worker_version": WORKER_VERSION
        }

    dbx = get_dropbox_client(data)

    if dbx is None:
        return {
            "error": (
                "Missing Dropbox credentials. "
                "Set DROPBOX_APP_KEY, "
                "DROPBOX_APP_SECRET, and "
                "DROPBOX_REFRESH_TOKEN."
            ),
            "worker_version": WORKER_VERSION
        }

    try:
        dbx.users_get_current_account()
        log("Dropbox authentication verified.")

    except Exception as error:
        log(f"Dropbox authentication failed: {error}")

        return {
            "error": f"Dropbox authentication failed: {error}",
            "worker_version": WORKER_VERSION
        }

    try:
        with tempfile.TemporaryDirectory() as temp_directory:
            input_zip = os.path.join(
                temp_directory,
                "input.zip"
            )

            output_zip = os.path.join(
                temp_directory,
                "results.zip"
            )

            input_directory = os.path.join(
                temp_directory,
                "input"
            )

            output_directory = os.path.join(
                temp_directory,
                "output"
            )

            os.makedirs(
                input_directory,
                exist_ok=True
            )

            os.makedirs(
                output_directory,
                exist_ok=True
            )

            dropbox_download(
                dbx,
                input_zip_dropbox_path,
                input_zip
            )

            unzip_file(
                input_zip,
                input_directory
            )

            input_files = list_files(
                input_directory
            )

            input_extension_counts = count_extensions(
                input_files
            )

            log(
                f"Input files found: "
                f"{len(input_files)}"
            )

            log(
                "Input extension counts: "
                f"{input_extension_counts}"
            )

            nef_files = [
                filename
                for filename in input_files
                if filename.lower().endswith(".nef")
            ]

            log(
                f"NEF files found: "
                f"{len(nef_files)}"
            )

            for filename in nef_files[:20]:
                log(f"NEF input: {filename}")

            log("Starting processor.")

            summary = process_mobile_job(
                input_directory,
                output_directory,
                job_config,
                progress_callback=None
            )

            log(
                "Processor complete. "
                f"Summary: {summary}"
            )

            output_files = list_files(
                output_directory
            )

            output_extension_counts = count_extensions(
                output_files
            )

            log(
                f"Output files created: "
                f"{len(output_files)}"
            )

            log(
                "Output extension counts: "
                f"{output_extension_counts}"
            )

            jpg_outputs = [
                filename
                for filename in output_files
                if filename.lower().endswith(
                    (".jpg", ".jpeg")
                )
            ]

            log(
                f"Display JPG outputs: "
                f"{len(jpg_outputs)}"
            )

            zip_folder(
                output_directory,
                output_zip
            )

            dropbox_upload(
                dbx,
                output_zip,
                output_zip_dropbox_path
            )

            log("Handler finished successfully.")

            return {
                "worker_version": WORKER_VERSION,
                "summary": summary,
                "output_zip_dropbox_path": (
                    output_zip_dropbox_path
                ),
                "rawpy": rawpy_status,
                "input_file_count": len(input_files),
                "input_extension_counts": (
                    input_extension_counts
                ),
                "nef_files_found": len(nef_files),
                "nef_files_sample": nef_files[:20],
                "output_file_count": len(output_files),
                "output_extension_counts": (
                    output_extension_counts
                ),
                "jpg_output_count": len(jpg_outputs)
            }

    except Exception as error:
        log(
            f"Handler failed: "
            f"{type(error).__name__}: {error}"
        )

        return {
            "error": (
                f"{type(error).__name__}: {error}"
            ),
            "worker_version": WORKER_VERSION
        }


runpod.serverless.start({
    "handler": handler
})
