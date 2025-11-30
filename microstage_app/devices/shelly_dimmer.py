from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests


def _normalize_base_url(host: str) -> str:
    host = host.strip()
    parsed = urlparse(host)
    if parsed.scheme:
        # host already includes scheme; keep only netloc/path to avoid duplicating
        base = f"{parsed.scheme}://{parsed.netloc or parsed.path}"
    else:
        base = f"http://{host}"
    return base.rstrip("/")


@dataclass
class ShellyState:
    on: bool
    brightness: int

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ShellyState":
        brightness = int(payload.get("brightness", 0))
        brightness = max(0, min(100, brightness))
        on = bool(payload.get("ison") or payload.get("on"))
        return cls(on=on, brightness=brightness)


class ShellyDimmer:
    """Lightweight wrapper around Shelly Dimmer 2 HTTP endpoints."""

    def __init__(self, host: str, *, timeout: float = 5.0, session: Optional[requests.Session] = None):
        if not host:
            raise ValueError("ShellyDimmer host must be provided")
        self.host = host
        self.base_url = _normalize_base_url(host)
        self.timeout = timeout
        self.session = session or requests.Session()

    # ------------------------------------------------------------------
    def connect(self) -> ShellyState:
        """Validate connectivity and return the current device state."""

        return self.health_check()

    def health_check(self) -> ShellyState:
        return self.get_status()

    def get_status(self) -> ShellyState:
        payload = self._request("/light/0")
        return ShellyState.from_payload(payload)

    def set_on(self, on: bool) -> ShellyState:
        params = {"turn": "on" if on else "off"}
        self._request("/light/0", params=params)
        return self.get_status()

    def set_brightness(self, brightness: int) -> ShellyState:
        value = max(0, min(100, int(brightness)))
        params = {"brightness": value}
        self._request("/light/0", params=params)
        return self.get_status()

    # ------------------------------------------------------------------
    def _request(self, path: str, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
