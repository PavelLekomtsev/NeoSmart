"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
from neosmart.config import reload_settings


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> None:
    """Drop the settings cache between tests so env/monkeypatch changes apply."""
    reload_settings()
