"""Credential-safe formatting for camera diagnostic output.

This module never loads camera configuration itself. Callers provide values
from the already-authorized credential source; tests use synthetic markers.
"""

from __future__ import annotations

import base64
import os
import re
from typing import Any, Iterable
from urllib.parse import quote, quote_plus, unquote, unquote_plus, urlsplit


_RTSP_URI = re.compile(r"(?i)\brtsps?://[^\s\"'<>]+")
_URI_SENTINEL = "\x01CAM_SIX_RTSP_URI\x02"


def child_environment_without_camera_secrets(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return an environment suitable for local tools that need no RTSP auth."""
    result = dict(os.environ if env is None else env)
    for key in tuple(result):
        normalized = key.upper()
        if (normalized.endswith("_RTSP_USERNAME") or normalized.endswith("_RTSP_PASSWORD")
                or normalized in {"RTSP_USERNAME", "RTSP_PASSWORD"}):
            result.pop(key, None)
    return result


class CredentialRedactor:
    """Redact known credential forms and all RTSP URLs from diagnostic text."""

    def __init__(self, secrets: Iterable[Any] = ()) -> None:
        self._raw: set[str] = set()
        self._encoded: set[str] = set()
        for secret in secrets:
            self.add_secret(secret)

    def add_secret(self, value: Any) -> None:
        if not isinstance(value, str) or not value:
            return
        self._raw.add(value)
        # URI encoders emit either hexadecimal case. Match these forms
        # case-insensitively during replacement.
        for encoded in (quote(value, safe=""), quote_plus(value, safe="")):
            if encoded and encoded != value:
                self._encoded.add(encoded)
        self._encoded.add(base64.b64encode(value.encode("utf-8")).decode("ascii"))

    def add_credentials(self, *, username: Any = None, password: Any = None,
                        uri: Any = None) -> None:
        user = username if isinstance(username, str) else ""
        password_text = password if isinstance(password, str) else ""
        uri_text = uri if isinstance(uri, str) else ""
        user_forms = {user} if user else set()
        password_forms = {password_text} if password_text else set()
        for value in tuple(user_forms):
            user_forms.update((unquote(value), unquote_plus(value)))
        for value in tuple(password_forms):
            password_forms.update((unquote(value), unquote_plus(value)))
        for value in user_forms | password_forms:
            self.add_secret(value)
        self.add_secret(uri_text)
        for user_value in user_forms:
            for password_value in password_forms:
                self.add_secret(f"{user_value}:{password_value}")
        # Camera config may provide credentials separately or embedded in its
        # URI. Add decoded userinfo forms without retaining or displaying them.
        if uri_text:
            try:
                parsed = urlsplit(uri_text)
                if parsed.username:
                    decoded_user = unquote(parsed.username)
                    self.add_secret(decoded_user)
                    self.add_secret(unquote_plus(parsed.username))
                if parsed.password:
                    decoded_password = unquote(parsed.password)
                    self.add_secret(decoded_password)
                    self.add_secret(unquote_plus(parsed.password))
                    if parsed.username:
                        for user_value in (decoded_user, unquote_plus(parsed.username)):
                            for password_value in (decoded_password, unquote_plus(parsed.password)):
                                self.add_secret(f"{user_value}:{password_value}")
            except (ValueError, UnicodeError):
                # Malformed URI metadata is still protected by whole-URI
                # masking and exact-value masking.
                pass

    @staticmethod
    def _replace_literal(text: str, secret: str) -> str:
        if not secret:
            return text
        if len(secret) <= 2 and secret.isalnum():
            # Avoid destroying unrelated diagnostics for tiny usernames while
            # still masking the standalone credential token.
            return re.sub(rf"(?<!\w){re.escape(secret)}(?!\w)", "<redacted>", text)
        return text.replace(secret, "<redacted>")

    def sanitize(self, value: Any) -> str:
        text = value if isinstance(value, str) else str(value)
        # A complete source URI is sensitive even if the configured URI has
        # no userinfo; do not persist endpoint or stream-path details.
        text = _RTSP_URI.sub(_URI_SENTINEL, text)
        for secret in sorted(self._raw, key=len, reverse=True):
            text = self._replace_literal(text, secret)
        for secret in sorted(self._encoded, key=len, reverse=True):
            if not secret:
                continue
            text = re.sub(re.escape(secret), "<redacted>", text, flags=re.IGNORECASE)
        return text.replace(_URI_SENTINEL, "<redacted-rtsp-uri>")

    def sanitize_exception(self, exc: BaseException) -> str:
        # Do not print traceback locals: those may contain the original
        # credential-bearing camera objects or subprocess buffers.
        return self.sanitize(f"{type(exc).__name__}: {exc}")

    def sanitize_args(self, args: Iterable[Any]) -> list[str]:
        return [self.sanitize(arg) for arg in args]

    def sanitize_child_output(self, output: Any) -> str:
        return self.sanitize(output)
