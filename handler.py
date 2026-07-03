import os
import zipfile
import tempfile
import runpod
import dropbox

from processor import process_mobile_job


def get_dropbox_client(data):
    refresh_token = data.get("dropbox_refresh_token", "")
    app_key = data.get("dropbox_app_key", "")
    app_secret = data.get("dropbox_app_secret", "")

    if not refresh_token:
        raise RuntimeError("Missing dropbox_refresh_token")

    if not app_key:
        raise RuntimeError("Missing dropbox_app_key")

    if not app_secret:
        raise RuntimeError("Missing dropbox_app_secret")

    return dropbox.Dropbox(
        oauth2_refresh_token=refresh_token.strip(),
        app_key=app_key.strip(),
        app_secret=app_secret.strip(),
    )


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


def dropbox_download(dbx, dropbox_path, local_path):
    _, res = dbx.files_download(dropbox_path)

    with open(local_path, "wb") as f:
        f.write(res.content)


def dropbox_upload(dbx, local_path, dropbox_path):
    with open(local_path, "rb") as f:
        dbx.files_upload(
            f.read(),
            dropbox_path,
            mode=dropbox.files.WriteMode.overwrite
        )


def handler(job):
    data = job.get("input", {})

    input_zip_dropbox_path = data.get("input_zip_dropbox_path")
    output_zip_dropbox_path = data.get("output_zip_dropbox_path")
    job_config = data.get("job_config", {})

    if not input_zip_dropbox_path:
        return {"error": "Missing input_zip_dropbox_path"}

    if not output_zip_dropbox_path:
        return {"error": "Missing output_zip_dropbox_path"}

    dbx = get_dropbox_client(data)

    with tempfile.TemporaryDirectory() as tmp:
        input_zip = os.path.join(tmp, "input.zip")
        output_zip = os.path.join(tmp, "results.zip")
        input_dir = os.path.join(tmp, "input")
        output_dir = os.path.join(tmp, "output")

        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        dropbox_download(dbx, input_zip_dropbox_path, input_zip)
        unzip_file(input_zip, input_dir)

        summary = process_mobile_job(
            input_dir,
            output_dir,
            job_config,
            progress_callback=None
        )

        zip_folder(output_dir, output_zip)
        dropbox_upload(dbx, output_zip, output_zip_dropbox_path)

        return {
            "summary": summary,
            "output_zip_dropbox_path": output_zip_dropbox_path
        }


runpod.serverless.start({"handler": handler})
