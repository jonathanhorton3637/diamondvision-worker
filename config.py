"""Configuration loading and validation for DiamondVision Worker 3.0."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Mapping


@dataclass(frozen=True)
class DropboxCredentials:
    """Dropbox authentication credentials."""

    access_token: str = ""
    app_key: str = ""
    app_secret: str = ""
    refresh_token: str = ""

    @property
    def has_refresh_credentials(self) -> bool:
        """Return True when all refresh-token credentials are available."""
        return bool(
            self.app_key
            and self.app_secret
            and self.refresh_token
        )

    @property
    def has_access_token(self) -> bool:
        """Return True when a direct access token is available."""
        return bool(self.access_token)

    @property
    def is_configured(self) -> bool:
        """Return True when either supported authentication method is configured."""
        return self.has_refresh_credentials or self.has_access_token


@dataclass(frozen=True)
class WorkerConfig:
    """Validated configuration for one DiamondVision RunPod job."""

    dropbox: DropboxCredentials
    input_zip_dropbox_path: str
    output_zip_dropbox_path: str
    job_config: dict[str, Any] = field(default_factory=dict)


def clean_value(value: Any) -> str:
    """Convert a configuration value to a stripped string."""
    if value is None:
        return ""

    return str(value).strip()


def get_first_value(
    data: Mapping[str, Any],
    payload_key: str,
    environment_key: str,
) -> str:
    """Return a payload value first, falling back to an environment variable."""
    payload_value = clean_value(data.get(payload_key))

    if payload_value:
        return payload_value

    return clean_value(os.environ.get(environment_key, ""))


def get_dropbox_credentials(
    data: Mapping[str, Any] | None = None,
) -> DropboxCredentials:
    """Load Dropbox credentials from job input and environment variables."""
    payload = data or {}

    return DropboxCredentials(
        access_token=get_first_value(
            payload,
            "dropbox_access_token",
            "DROPBOX_ACCESS_TOKEN",
        ),
        app_key=get_first_value(
            payload,
            "dropbox_app_key",
            "DROPBOX_APP_KEY",
        ),
        app_secret=get_first_value(
            payload,
            "dropbox_app_secret",
            "DROPBOX_APP_SECRET",
        ),
        refresh_token=get_first_value(
            payload,
            "dropbox_refresh_token",
            "DROPBOX_REFRESH_TOKEN",
        ),
    )


def get_required_string(
    data: Mapping[str, Any],
    key: str,
) -> str:
    """Return a required non-empty string from the payload."""
    value = clean_value(data.get(key))

    if not value:
        raise ValueError(f"Missing required input: {key}")

    return value


def get_optional_mapping(
    data: Mapping[str, Any],
    key: str,
) -> dict[str, Any]:
    """Return an optional dictionary payload value."""
    value = data.get(key)

    if value is None:
        return {}

    if not isinstance(value, Mapping):
        raise ValueError(f"Input '{key}' must be a JSON object.")

    return dict(value)


def validate_dropbox_credentials(
    credentials: DropboxCredentials,
) -> None:
    """Validate that a complete Dropbox authentication method is available."""
    if credentials.is_configured:
        return

    raise ValueError(
        "Dropbox authentication is not configured. Provide either "
        "DROPBOX_ACCESS_TOKEN or all of DROPBOX_APP_KEY, "
        "DROPBOX_APP_SECRET, and DROPBOX_REFRESH_TOKEN."
    )


def load_config(
    data: Mapping[str, Any] | None = None,
) -> WorkerConfig:
    """Load and validate Worker 3.0 configuration."""
    payload = dict(data or {})

    credentials = get_dropbox_credentials(payload)
    validate_dropbox_credentials(credentials)

    input_zip_dropbox_path = get_required_string(
        payload,
        "input_zip_dropbox_path",
    )
    output_zip_dropbox_path = get_required_string(
        payload,
        "output_zip_dropbox_path",
    )
    job_config = get_optional_mapping(
        payload,
        "job_config",
    )

    return WorkerConfig(
        dropbox=credentials,
        input_zip_dropbox_path=input_zip_dropbox_path,
        output_zip_dropbox_path=output_zip_dropbox_path,
        job_config=job_config,
    )
