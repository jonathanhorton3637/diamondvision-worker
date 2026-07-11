"""Dropbox transfer utilities for DiamondVision Worker 3.0."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

import dropbox
from dropbox.files import (
    CommitInfo,
    FileMetadata,
    UploadSessionCursor,
    WriteMode,
)

from config import DropboxCredentials, WorkerConfig


LOGGER = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024
SMALL_UPLOAD_LIMIT = 140 * 1024 * 1024
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_BASE_DELAY = 2.0
DEFAULT_TIMEOUT_SECONDS = 300


class DropboxConfigurationError(RuntimeError):
    """Raised when Dropbox authentication is not configured."""


class DropboxTransferError(RuntimeError):
    """Raised when a Dropbox transfer cannot be completed."""


def normalize_dropbox_path(path: str) -> str:
    """Return a normalized absolute Dropbox path."""
    normalized = str(path or "").strip().replace("\\", "/")

    if not normalized:
        raise ValueError("Dropbox path cannot be empty.")

    if not normalized.startswith("/"):
        normalized = f"/{normalized}"

    while "//" in normalized:
        normalized = normalized.replace("//", "/")

    if len(normalized) > 1:
        normalized = normalized.rstrip("/")

    return normalized


def retry_operation(
    operation_name: str,
    operation: Callable[[], T],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
) -> T:
    """Run an operation with exponential backoff and jitter."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")

    if base_delay < 0:
        raise ValueError("base_delay cannot be negative.")

    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()

        except Exception as error:
            last_error = error

            LOGGER.warning(
                "%s failed on attempt %s/%s: %s: %s",
                operation_name,
                attempt,
                max_attempts,
                type(error).__name__,
                error,
            )

            if attempt >= max_attempts:
                break

            exponential_delay = base_delay * (2 ** (attempt - 1))
            jitter = random.uniform(0.0, 1.0)
            delay = exponential_delay + jitter

            LOGGER.info(
                "Retrying %s in %.1f seconds.",
                operation_name,
                delay,
            )

            time.sleep(delay)

    if last_error is None:
        raise DropboxTransferError(
            f"{operation_name} failed without returning an error."
        )

    raise DropboxTransferError(
        f"{operation_name} failed after {max_attempts} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


def extract_credentials(
    config: WorkerConfig | DropboxCredentials,
) -> DropboxCredentials:
    """Extract Dropbox credentials from a supported configuration object."""
    if isinstance(config, DropboxCredentials):
        return config

    if isinstance(config, WorkerConfig):
        return config.dropbox

    raise TypeError(
        "Expected WorkerConfig or DropboxCredentials, "
        f"received {type(config).__name__}."
    )


def create_dropbox_client(
    config: WorkerConfig | DropboxCredentials,
) -> dropbox.Dropbox:
    """Create an authenticated Dropbox SDK client."""
    credentials = extract_credentials(config)

    if credentials.has_refresh_credentials:
        LOGGER.info("Using Dropbox refresh-token authentication.")

        return dropbox.Dropbox(
            oauth2_refresh_token=credentials.refresh_token,
            app_key=credentials.app_key,
            app_secret=credentials.app_secret,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )

    if credentials.has_access_token:
        LOGGER.info("Using Dropbox access-token authentication.")

        return dropbox.Dropbox(
            oauth2_access_token=credentials.access_token,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )

    raise DropboxConfigurationError(
        "Missing Dropbox credentials. Configure either "
        "DROPBOX_ACCESS_TOKEN or all of DROPBOX_APP_KEY, "
        "DROPBOX_APP_SECRET, and DROPBOX_REFRESH_TOKEN."
    )


def verify_dropbox_client(
    client: dropbox.Dropbox,
) -> dict[str, str]:
    """Verify Dropbox authentication and return basic account information."""
    account = retry_operation(
        "Dropbox authentication verification",
        client.users_get_current_account,
    )

    account_name = ""

    try:
        account_name = account.name.display_name
    except Exception:
        LOGGER.debug(
            "Dropbox account display name was unavailable.",
            exc_info=True,
        )

    LOGGER.info(
        "Dropbox authentication verified%s.",
        f" for {account_name}" if account_name else "",
    )

    return {
        "account_id": str(getattr(account, "account_id", "")),
        "display_name": account_name,
    }


def ensure_dropbox_parent_folder(
    client: dropbox.Dropbox,
    dropbox_path: str,
) -> None:
    """Create all missing parent folders for a Dropbox path."""
    normalized_path = normalize_dropbox_path(dropbox_path)
    parent = os.path.dirname(normalized_path)

    if not parent or parent == "/":
        return

    current = ""

    for part in parent.strip("/").split("/"):
        if not part:
            continue

        current = f"{current}/{part}"

        try:
            retry_operation(
                f"Create Dropbox folder {current}",
                lambda current_path=current: client.files_create_folder_v2(
                    current_path,
                    autorename=False,
                ),
            )

            LOGGER.info("Created Dropbox folder: %s", current)

        except DropboxTransferError as error:
            cause = error.__cause__

            if isinstance(cause, dropbox.exceptions.ApiError):
                api_error = cause.error

                if (
                    api_error.is_path()
                    and api_error.get_path().is_conflict()
                ):
                    LOGGER.debug(
                        "Dropbox folder already exists: %s",
                        current,
                    )
                    continue

            raise


def get_remote_metadata(
    client: dropbox.Dropbox,
    dropbox_path: str,
) -> FileMetadata:
    """Fetch metadata for a Dropbox file."""
    normalized_path = normalize_dropbox_path(dropbox_path)

    metadata = retry_operation(
        f"Read Dropbox metadata {normalized_path}",
        lambda: client.files_get_metadata(normalized_path),
    )

    if not isinstance(metadata, FileMetadata):
        raise DropboxTransferError(
            f"Dropbox path is not a file: {normalized_path}"
        )

    return metadata


def download_file(
    client: dropbox.Dropbox,
    dropbox_path: str,
    local_path: str,
) -> None:
    """Download a Dropbox file with retry and size verification."""
    normalized_path = normalize_dropbox_path(dropbox_path)
    local = Path(local_path)

    local.parent.mkdir(parents=True, exist_ok=True)

    if local.exists():
        local.unlink()

    LOGGER.info(
        "Downloading Dropbox file %s to %s.",
        normalized_path,
        local,
    )

    def perform_download() -> None:
        client.files_download_to_file(
            str(local),
            normalized_path,
        )

    retry_operation(
        f"Download {normalized_path}",
        perform_download,
    )

    if not local.is_file():
        raise DropboxTransferError(
            f"Dropbox download did not create the local file: {local}"
        )

    metadata = get_remote_metadata(client, normalized_path)
    local_size = local.stat().st_size
    remote_size = int(metadata.size)

    if local_size != remote_size:
        raise DropboxTransferError(
            "Downloaded file size does not match Dropbox metadata. "
            f"Local: {local_size:,} bytes. "
            f"Remote: {remote_size:,} bytes."
        )

    LOGGER.info(
        "Download complete: %s bytes.",
        f"{local_size:,}",
    )


def upload_small_file(
    client: dropbox.Dropbox,
    local_path: str,
    dropbox_path: str,
) -> FileMetadata:
    """Upload a file using Dropbox's single-request upload API."""
    normalized_path = normalize_dropbox_path(dropbox_path)

    def perform_upload() -> FileMetadata:
        with open(local_path, "rb") as file_handle:
            result = client.files_upload(
                file_handle.read(),
                normalized_path,
                mode=WriteMode.overwrite,
                autorename=False,
                mute=True,
                strict_conflict=False,
            )

        if not isinstance(result, FileMetadata):
            raise DropboxTransferError(
                f"Dropbox did not return file metadata for {normalized_path}."
            )

        return result

    return retry_operation(
        f"Upload {normalized_path}",
        perform_upload,
    )


def start_upload_session(
    client: dropbox.Dropbox,
    first_chunk: bytes,
) -> str:
    """Start a Dropbox upload session."""
    result = retry_operation(
        "Start Dropbox upload session",
        lambda: client.files_upload_session_start(
            first_chunk,
            close=False,
        ),
    )

    session_id = str(getattr(result, "session_id", ""))

    if not session_id:
        raise DropboxTransferError(
            "Dropbox upload session did not return a session ID."
        )

    return session_id


def append_upload_session(
    client: dropbox.Dropbox,
    chunk: bytes,
    cursor: UploadSessionCursor,
) -> None:
    """Append one non-final chunk to a Dropbox upload session."""
    retry_operation(
        f"Append upload session at offset {cursor.offset:,}",
        lambda: client.files_upload_session_append_v2(
            chunk,
            cursor,
            close=False,
        ),
    )


def finish_upload_session(
    client: dropbox.Dropbox,
    chunk: bytes,
    cursor: UploadSessionCursor,
    commit: CommitInfo,
) -> FileMetadata:
    """Finish a Dropbox upload session and commit the file."""
    result = retry_operation(
        f"Finish upload session at offset {cursor.offset:,}",
        lambda: client.files_upload_session_finish(
            chunk,
            cursor,
            commit,
        ),
    )

    if not isinstance(result, FileMetadata):
        raise DropboxTransferError(
            "Dropbox upload session did not return file metadata."
        )

    return result


def upload_large_file(
    client: dropbox.Dropbox,
    local_path: str,
    dropbox_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> FileMetadata:
    """Upload a large file through a Dropbox upload session."""
    normalized_path = normalize_dropbox_path(dropbox_path)
    file_size = os.path.getsize(local_path)

    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than zero.")

    if file_size == 0:
        return upload_small_file(
            client,
            local_path,
            normalized_path,
        )

    commit = CommitInfo(
        path=normalized_path,
        mode=WriteMode.overwrite,
        autorename=False,
        mute=True,
        strict_conflict=False,
    )

    with open(local_path, "rb") as file_handle:
        first_chunk = file_handle.read(chunk_size)

        if not first_chunk:
            raise DropboxTransferError(
                "Could not read the first upload chunk."
            )

        session_id = start_upload_session(
            client,
            first_chunk,
        )

        cursor = UploadSessionCursor(
            session_id=session_id,
            offset=len(first_chunk),
        )

        LOGGER.info(
            "Dropbox upload session started at offset %s/%s.",
            f"{cursor.offset:,}",
            f"{file_size:,}",
        )

        if cursor.offset >= file_size:
            return finish_upload_session(
                client,
                b"",
                cursor,
                commit,
            )

        while cursor.offset < file_size:
            remaining = file_size - cursor.offset
            read_size = min(chunk_size, remaining)
            chunk = file_handle.read(read_size)

            if not chunk:
                raise DropboxTransferError(
                    "Unexpected end of local file during Dropbox upload."
                )

            final_chunk = cursor.offset + len(chunk) >= file_size

            if final_chunk:
                result = finish_upload_session(
                    client,
                    chunk,
                    cursor,
                    commit,
                )

                cursor.offset += len(chunk)

            else:
                append_upload_session(
                    client,
                    chunk,
                    cursor,
                )

                cursor.offset += len(chunk)
                result = None

            progress = (cursor.offset / file_size) * 100.0

            LOGGER.info(
                "Dropbox upload progress: %s/%s bytes (%.1f%%).",
                f"{cursor.offset:,}",
                f"{file_size:,}",
                progress,
            )

            if result is not None:
                return result

    raise DropboxTransferError(
        "Dropbox upload session ended without committing the file."
    )


def verify_upload(
    client: dropbox.Dropbox,
    local_path: str,
    dropbox_path: str,
) -> bool:
    """Verify that a Dropbox upload exists and matches the local file size."""
    local = Path(local_path)

    if not local.is_file():
        raise FileNotFoundError(
            f"Upload verification source does not exist: {local}"
        )

    metadata = get_remote_metadata(
        client,
        dropbox_path,
    )

    local_size = local.stat().st_size
    remote_size = int(metadata.size)

    if local_size != remote_size:
        raise DropboxTransferError(
            "Uploaded file size does not match the local file. "
            f"Local: {local_size:,} bytes. "
            f"Dropbox: {remote_size:,} bytes."
        )

    LOGGER.info(
        "Dropbox upload verified: %s bytes.",
        f"{local_size:,}",
    )

    return True


def upload_file(
    client: dropbox.Dropbox,
    local_path: str,
    dropbox_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    """Upload and verify a local file in Dropbox."""
    local = Path(local_path)

    if not local.is_file():
        raise FileNotFoundError(
            f"Upload source does not exist: {local}"
        )

    normalized_path = normalize_dropbox_path(dropbox_path)

    ensure_dropbox_parent_folder(
        client,
        normalized_path,
    )

    file_size = local.stat().st_size

    LOGGER.info(
        "Uploading %s to %s.",
        local,
        normalized_path,
    )
    LOGGER.info(
        "Upload size: %s bytes.",
        f"{file_size:,}",
    )

    if file_size <= SMALL_UPLOAD_LIMIT:
        metadata = upload_small_file(
            client,
            str(local),
            normalized_path,
        )
        upload_method = "single_request"

    else:
        metadata = upload_large_file(
            client,
            str(local),
            normalized_path,
            chunk_size=chunk_size,
        )
        upload_method = "upload_session"

    verify_upload(
        client,
        str(local),
        normalized_path,
    )

    LOGGER.info("Dropbox upload complete.")

    return {
        "dropbox_path": normalized_path,
        "file_name": metadata.name,
        "file_id": metadata.id,
        "revision": metadata.rev,
        "size": int(metadata.size),
        "content_hash": str(metadata.content_hash or ""),
        "upload_method": upload_method,
        "verified": True,
    }
