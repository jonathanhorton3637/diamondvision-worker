import os
import zipfile
import tempfile
import base64
import runpod

from processor import process_mobile_job


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


def handler(job):
    data = job["input"]

    photos_b64 = data.get("photos_zip_base64", "")
    job_config = data.get("job_config", {})

    if not photos_b64:
        return {"error": "Missing photos_zip_base64"}

    with tempfile.TemporaryDirectory() as tmp:
        input_zip = os.path.join(tmp, "input.zip")
        input_dir = os.path.join(tmp, "input")
        output_dir = os.path.join(tmp, "output")
        output_zip = os.path.join(tmp, "output.zip")

        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        with open(input_zip, "wb") as f:
            f.write(base64.b64decode(photos_b64))

        unzip_file(input_zip, input_dir)

        summary = process_mobile_job(
            input_dir,
            output_dir,
            job_config,
            progress_callback=None
        )

        zip_folder(output_dir, output_zip)

        with open(output_zip, "rb") as f:
            result_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "summary": summary,
            "output_zip_base64": result_b64
        }


runpod.serverless.start({"handler": handler})
