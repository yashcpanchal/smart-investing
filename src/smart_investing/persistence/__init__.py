"""State persistence (strategies, proposals, portfolio, audit)."""

from smart_investing.persistence.repo import StateRepo, default_state_path

__all__ = ["StateRepo", "default_state_path"]
