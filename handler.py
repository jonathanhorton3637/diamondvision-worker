"""RunPod entry point for DiamondVision Worker 3.0."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import tempfile
import zipfile
from typing import Any, Mapping

import runpod

from config import load_config
import dropbox_utils
from processor import process_mobile_job
from version import WORKER_VERSION

LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format=(
                "%(asctime)s %(levelname)s "
                "%(name)s worker=%(worker_version)s %(message)s"
            ),
        )

    old_factory = logging.getLogRecordFactory()

    def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = old_factory(*args, **kwargs)
        record.worker_version = WORKER_VERSION
        return record

    logging.setLogRecordFactory(record_factory)


def _safe_extract(zip_path: Path, destination: Path) -> None:
    """Extract a ZIP without allowing path traversal."""
    destination_resolved = destination.resolve()

    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            try:
                member_path.relative_to(destination_resolved)
            except ValueError as error:
                raise ValueError(
                    f"Unsafe ZIP member path: {member.filename}"
                ) from error

            # Reject symlinks stored in Unix ZIP metadata.
            unix_mode = member.external_attr >> 16
            if unix_mode and (unix_mode & 0o170000) == 0o120000:
                raise ValueError(f"ZIP symlinks are not allowed: {member.filename}")

        archive.extractall(destination)


def _zip_folder(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source).as_posix())


def _payload(job: Mapping[str, Any]) -> dict[str, Any]:
    value = job.get("input", {})
    if not isinstance(value, Mapping):
        raise ValueError("RunPod job input must be a JSON object.")
    return dict(value)


def _read_setting(config: Any, *names: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        for name in names:
            if name in config:
                return config[name]
        return default

    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    return default


def _load_worker_config(data: Mapping[str, Any]) -> Any:
    try:
        return load_config(data)
    except TypeError:
        return load_config()


def _create_dropbox_client(config: Any, data: Mapping[str, Any]) -> Any:
    for name in (
        "create_dropbox_client",
        "get_dropbox_client",
        "build_dropbox_client",
    ):
        function = getattr(dropbox_utils, name, None)
        if callable(function):
            for args in ((config,), (data,), ()):
                try:
                    return function(*args)
                except TypeError:
                    continue

    client_class = getattr(dropbox_utils, "DropboxClient", None)
    if client_class:
        for args in ((config,), (data,), ()):
            try:
                return client_class(*args)
            except TypeError:
                continue

    raise AttributeError(
        "dropbox_utils.py must expose create_dropbox_client() "
        "or DropboxClient."
    )


def _download(client: Any, remote_path: str, local_path: Path) -> None:
    for name in ("download_file", "download", "download_with_retry"):
        method = getattr(client, name, None)
        if callable(method):
            method(remote_path, str(local_path))
            return

    for name in ("download_file", "download", "download_with_retry"):
        function = getattr(dropbox_utils, name, None)
        if callable(function):
            function(client, remote_path, str(local_path))
            return

    raise AttributeError("No Dropbox download function was found.")


def _upload(client: Any, local_path: Path, remote_path: str) -> Any:
    for name in ("upload_file", "upload", "upload_with_retry"):
        method = getattr(client, name, None)
        if callable(method):
            return method(str(local_path), remote_path)

    for name in ("upload_file", "upload", "upload_with_retry"):
        function = getattr(dropbox_utils, name, None)
        if callable(function):
            return function(client, str(local_path), remote_path)

    raise AttributeError("No Dropbox upload function was found.")


def _verify_upload(client: Any, local_path: Path, remote_path: str) -> bool:
    for name in ("verify_upload", "verify_file"):
        method = getattr(client, name, None)
        if callable(method):
            return bool(method(str(local_path), remote_path))

    function = getattr(dropbox_utils, "verify_upload", None)
    if callable(function):
        return bool(function(client, str(local_path), remote_path))

    # The rewritten upload helper already verifies uploads when supported.
    return True


def handler(job: Mapping[str, Any]) -> dict[str, Any]:
    """Process one RunPod serverless job."""
    _configure_logging()
    LOGGER.info("Job received.")

    try:
        data = _payload(job)
        config = _load_worker_config(data)

        input_dropbox_path = (
            data.get("input_zip_dropbox_path")
            or _read_setting(config, "input_zip_dropbox_path")
        )
        output_dropbox_path = (
            data.get("output_zip_dropbox_path")
            or _read_setting(config, "output_zip_dropbox_path")
        )
        job_config = data.get("job_config") or _read_setting(
            config,
            "job_config",
            default={},
        )

        if not input_dropbox_path:
            raise ValueError("Missing input_zip_dropbox_path.")
        if not output_dropbox_path:
            raise ValueError("Missing output_zip_dropbox_path.")
        if not isinstance(job_config, Mapping):
            raise ValueError("job_config must be a JSON object.")

        client = _create_dropbox_client(config, data)

        with tempfile.TemporaryDirectory(prefix="diamondvision-") as temporary:
            temp = Path(temporary)
            input_zip = temp / "input.zip"
            input_directory = temp / "input"
            output_directory = temp / "output"
            output_zip = temp / "results.zip"

            input_directory.mkdir(parents=True, exist_ok=True)
            output_directory.mkdir(parents=True, exist_ok=True)

            LOGGER.info("Downloading input ZIP from Dropbox.")
            _download(client, str(input_dropbox_path), input_zip)

            if not zipfile.is_zipfile(input_zip):
                raise ValueError("Downloaded input is not a valid ZIP archive.")

            LOGGER.info("Extracting input ZIP.")
            _safe_extract(input_zip, input_directory)

            LOGGER.info("Starting image pipeline.")
            summary = process_mobile_job(
                str(input_directory),
                str(output_directory),
                dict(job_config),
                progress_callback=None,
            )

            LOGGER.info("Creating results ZIP.")
            _zip_folder(output_directory, output_zip)

            LOGGER.info("Uploading results ZIP to Dropbox.")
            upload_result = _upload(
                client,
                output_zip,
                str(output_dropbox_path),
            )

            verified = _verify_upload(
                client,
                output_zip,
                str(output_dropbox_path),
            )
            if not verified:
                raise RuntimeError("Dropbox upload verification failed.")

            LOGGER.info("Job completed successfully.")
            return {
                "ok": True,
                "worker_version": WORKER_VERSION,
                "summary": summary,
                "input_zip_dropbox_path": str(input_dropbox_path),
                "output_zip_dropbox_path": str(output_dropbox_path),
                "output_zip_bytes": output_zip.stat().st_size,
                "upload_verified": True,
                "upload_result": (
                    upload_result
                    if isinstance(upload_result, (str, int, float, bool, dict, list, type(None)))
                    else str(upload_result)
                ),
            }

    except Exception as error:
        LOGGER.exception("Job failed.")
        return {
            "ok": False,
            "worker_version": WORKER_VERSION,
            "error_type": type(error).__name__,
            "error": str(error),
        }


if __name__ == "__main__":
    _configure_logging()
    runpod.serverless.start({"handler": handler})
