import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DropboxCredentials:
    access_token: str = ""
    app_key: str = ""
    app_secret: str = ""
    refresh_token: str = ""

    @property
    def has_refresh_credentials(self) -> bool:
        return bool(
            self.app_key
            and self.app_secret
            and self.refresh_token
        )

    @property
    def has_access_token(self) -> bool:
        return bool(self.access_token)

    @property
    def is_configured(self) -> bool:
        return (
            self.has_refresh_credentials
            or self.has_access_token
        )


def clean_value(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def get_first_value(
    data: dict,
    payload_key: str,
    environment_key: str,
) -> str:
    payload_value = clean_value(data.get(payload_key))

    if payload_value:
        return payload_value

    return clean_value(os.environ.get(environment_key, ""))


def get_dropbox_credentials(
    data: dict | None = None,
) -> DropboxCredentials:
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
    data: dict,
    key: str,
) -> str:
    value = clean_value(data.get(key))

    if not value:
        raise ValueError(f"Missing required input: {key}")

    return value
