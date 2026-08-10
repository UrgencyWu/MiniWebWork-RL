"""Gunicorn entrypoint wrapping the pinned Agent-R1 service safely."""

from __future__ import annotations

from recipes.webshop.env.server import app as upstream_app

from .serialization import ProcessSerializedASGI

app = ProcessSerializedASGI(upstream_app)
