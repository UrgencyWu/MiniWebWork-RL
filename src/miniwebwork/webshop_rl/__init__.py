"""Focused WebShop adapters for the M5/M6 studies.

HTTP runtime imports are lazy so the pure protocol, corpus and credit audits
remain runnable in CPU-only validation environments.
"""

from __future__ import annotations

from typing import Any

from .credit import ANCHOR_METHOD, BASELINE_METHOD, CREDIT_FORMULA_VERSION
from .actions import WebShopCommand, parse_command_output

__all__ = [
    "ANCHOR_METHOD",
    "BASELINE_METHOD",
    "CREDIT_FORMULA_VERSION",
    "WebShopCommand",
    "WebShopHTTPEnvironment",
    "WebShopObservation",
    "parse_command_output",
]


def __getattr__(name: str) -> Any:
    if name in {"WebShopHTTPEnvironment", "WebShopObservation"}:
        from .environment import WebShopHTTPEnvironment, WebShopObservation

        return {
            "WebShopHTTPEnvironment": WebShopHTTPEnvironment,
            "WebShopObservation": WebShopObservation,
        }[name]
    raise AttributeError(name)
