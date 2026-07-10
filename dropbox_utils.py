import os
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

import dropbox
from dropbox.files import (
    CommitInfo,
    UploadSessionCursor,
    WriteMode,
)

from config import DropboxCredentials
from version import WORKER_VERSION


T = TypeVar("T")

DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024
SMALL_UPLOAD_LIMIT = 140 * 1024 * 1024
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_BASE_DELAY = 2.0


def log(message: str) -> None:
    print(
        f"[{WORKER_VERSION}] [Dropbox] {message}",
        flush=True,
    )


class DropboxConfigurationError(RuntimeError):
    pass


class DropboxTransferError(RuntimeError):
    pass


def retry_operation(
    operation_name: str,
    operation: Callable[[], T],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
) -> T:
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()

        except Exception as error:
            last_error = error

            if attempt >= max_attempts:
                break

            exponential_delay = base_delay * (2 ** (attempt - 1))
            jitter = random.uniform(0, 1)
            delay = exponential_delay + jitter

            log(
                f"{operation_name} failed on attempt "
                f"{attempt}/{max_attempts}: "
                f"{type(error).__name__}: {error}"
            )

            log(
                f"Retrying in {delay:.1f} seconds."
            )

            time.sleep(delay)

    raise DropboxTransferError(
        f"{operation_name} failed after "
        f"{max_attempts} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


def create_dropbox_client(
    credentials: DropboxCredentials,
) -> dropbox.Dropbox:
    if credentials.has_refresh_credentials:
        log("Using refresh-token authentication.")

        return dropbox.Dropbox(
            oauth2_refresh_token=credentials.refresh_token,
            app_key=credentials.app_key,
            app_secret=credentials.app_secret,
            timeout=300,
        )

    if credentials.has_access_token:
        log("Using access-token authentication.")

        return dropbox.Dropbox(
            oauth2_access_token=credentials.access_token,
            timeout=300,
        )

    raise DropboxConfigurationError(
        "Missing Dropbox credentials. Configure either "
        "DROPBOX_ACCESS_TOKEN or all three refresh-token "
        "variables: DROPBOX_APP_KEY, DROPBOX_APP_SECRET, "
        "and DROPBOX_REFRESH_TOKEN."
    )


def verify_dropbox_client(
    client: dropbox.Dropbox,
) -> dict[str, Any]:
    account = retry_operation(
        "Dropbox authentication verification",
        client.users_get_current_account,
    )

    account_name = ""

    try:
        account_name = account.name.display_name
    except Exception:
        account_name = ""

    log(
        "Authentication verified"
        + (
            f" for {account_name}."
            if account_name
            else "."
        )
    )

    return {
        "account_id": getattr(
            account,
            "account_id",
            "",
        ),
        "display_name": account_name,
    }


def ensure_dropbox_parent_folder(
    client: dropbox.Dropbox,
    dropbox_path: str,
) -> None:
    normalized = dropbox_path.replace("\\", "/")
    parent = os.path.dirname(normalized)

    if not parent or parent == "/":
        return

    current = ""

    for part in parent.split("/"):
        if not part:
            continue

        current += f"/{part}"

        try:
            client.files_create_folder_v2(
                current,
                autorename=False,
            )

            log(f"Created Dropbox folder: {current}")

        except dropbox.exceptions.ApiError as error:
            if (
                error.error.is_path()
                and error.error.get_path().is_conflict()
            ):
                continue

            raise


def download_file(
    client: dropbox.Dropbox,
    dropbox_path: str,
    local_path: str,
) -> None:
    os.makedirs(
        os.path.dirname(local_path),
        exist_ok=True,
    )

    log(f"Downloading: {dropbox_path}")

    def perform_download() -> None:
        client.files_download_to_file(
            local_path,
            dropbox_path,
        )

    retry_operation(
        f"Download {dropbox_path}",
        perform_download,
    )

    if not os.path.exists(local_path):
        raise DropboxTransferError(
            f"Dropbox download did not create "
            f"the local file: {local_path}"
        )

    local_size = os.path.getsize(local_path)

    log(
        f"Download complete: {local_size:,} bytes."
    )


def upload_small_file(
    client: dropbox.Dropbox,
    local_path: str,
    dropbox_path: str,
) -> None:
    def perform_upload() -> None:
        with open(local_path, "rb") as file:
            client.files_upload(
                file.read(),
                dropbox_path,
                mode=WriteMode.overwrite,
                autorename=False,
                mute=True,
            )

    retry_operation(
        f"Small-file upload {dropbox_path}",
        perform_upload,
    )


def start_upload_session(
    client: dropbox.Dropbox,
    chunk: bytes,
) -> str:
    result = retry_operation(
        "Start upload session",
        lambda: client.files_upload_session_start(
            chunk,
            close=False,
        ),
    )

    return result.session_id


def append_upload_session(
    client: dropbox.Dropbox,
    chunk: bytes,
    cursor: UploadSessionCursor,
) -> None:
    retry_operation(
        f"Append upload session at offset "
        f"{cursor.offset:,}",
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
) -> None:
    retry_operation(
        f"Finish upload session at offset "
        f"{cursor.offset:,}",
        lambda: client.files_upload_session_finish(
            chunk,
            cursor,
            commit,
        ),
    )


def upload_large_file(
    client: dropbox.Dropbox,
    local_path: str,
    dropbox_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    file_size = os.path.getsize(local_path)

    if file_size == 0:
        upload_small_file(
            client,
            local_path,
            dropbox_path,
        )
        return

    commit = CommitInfo(
        path=dropbox_path,
        mode=WriteMode.overwrite,
        autorename=False,
        mute=True,
    )

    with open(local_path, "rb") as file:
        first_chunk = file.read(chunk_size)

        session_id = start_upload_session(
            client,
            first_chunk,
        )

        cursor = UploadSessionCursor(
            session_id=session_id,
            offset=len(first_chunk),
        )

        log(
            f"Upload session started. "
            f"Offset: {cursor.offset:,}/{file_size:,}"
        )

        while cursor.offset < file_size:
            remaining = file_size - cursor.offset
            read_size = min(chunk_size, remaining)
            chunk = file.read(read_size)

            if not chunk:
                raise DropboxTransferError(
                    "Unexpected end of file during "
                    "Dropbox upload session."
                )

            is_final_chunk = (
                cursor.offset + len(chunk)
                >= file_size
            )

            if is_final_chunk:
                finish_upload_session(
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

            progress = (
                cursor.offset / file_size
            ) * 100

            log(
                f"Upload progress: "
                f"{cursor.offset:,}/{file_size:,} "
                f"bytes ({progress:.1f}%)."
            )


def upload_file(
    client: dropbox.Dropbox,
    local_path: str,
    dropbox_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    if not os.path.isfile(local_path):
        raise FileNotFoundError(
            f"Upload source does not exist: "
            f"{local_path}"
        )

    ensure_dropbox_parent_folder(
        client,
        dropbox_path,
    )

    file_size = os.path.getsize(local_path)

    log(
        f"Uploading {local_path} to "
        f"{dropbox_path}."
    )

    log(f"Upload size: {file_size:,} bytes.")

    if file_size <= SMALL_UPLOAD_LIMIT:
        upload_small_file(
            client,
            local_path,
            dropbox_path,
        )
    else:
        upload_large_file(
            client,
            local_path,
            dropbox_path,
            chunk_size=chunk_size,
        )

    log("Upload complete.")
