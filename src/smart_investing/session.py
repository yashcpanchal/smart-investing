"""In-memory conversation sessions.

A session is the running state of one back-and-forth: the thesis, the tuning
knobs, the user-curated universe (pinned/excluded as they walk the graph and
give feedback), and the last built proposal + chat transcript. Ephemeral by
design — proposals themselves are persisted by StateRepo.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from smart_investing.domain.types import Proposal, StrategySpec


@dataclass
class Session:
    id: str
    theme: str = ""  # the user's thesis, as they phrased it (for display)
    search_theme: str = ""  # clean extracted themes, for retrieval + graph search
    base_spec: StrategySpec | None = None  # cached LLM theme-parse; reused across refine turns
    # tuning knobs (same vocabulary the clarify follow-ups use)
    answers: dict = field(default_factory=lambda: {"risk": "balanced", "breadth": "balanced", "supply_chain": "yes"})
    # evidence-source blend for universe ranking (defaults match SourceWeights)
    source_weights: dict = field(
        default_factory=lambda: {"sec_13f": 0.5, "insider": 0.2, "news_sentiment": 0.2, "social": 0.1}
    )
    lookback: str = "2y"
    cash: float = 10_000.0
    pinned: list[str] = field(default_factory=list)  # symbols the user explicitly added
    excluded: list[str] = field(default_factory=list)  # symbols the user removed
    proposal: Proposal | None = None
    messages: list[dict] = field(default_factory=list)  # [{role, text}]

    def pin(self, symbols: list[str]) -> list[str]:
        added = []
        for s in symbols:
            s = s.upper()
            if s in self.excluded:
                self.excluded.remove(s)
            if s not in self.pinned:
                self.pinned.append(s)
                added.append(s)
        return added

    def exclude(self, symbols: list[str]) -> list[str]:
        removed = []
        for s in symbols:
            s = s.upper()
            if s in self.pinned:
                self.pinned.remove(s)
            if s not in self.excluded:
                self.excluded.append(s)
                removed.append(s)
        return removed

    def compile_answers(self) -> dict:
        """The answers dict passed to compile_strategy, folding in pin/exclude."""
        return {
            **self.answers,
            "include_symbols": list(self.pinned),
            "exclude_symbols": list(self.excluded),
            "source_weights": dict(self.source_weights),
        }

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "theme": self.theme,
            "search_theme": self.search_theme or self.theme,
            "risk": self.answers.get("risk"),
            "breadth": self.answers.get("breadth"),
            "supply_chain": self.answers.get("supply_chain"),
            "lookback": self.lookback,
            "cash": self.cash,
            "pinned": list(self.pinned),
            "excluded": list(self.excluded),
            "source_weights": dict(self.source_weights),
        }


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self) -> Session:
        sid = uuid.uuid4().hex[:12]
        s = Session(id=sid)
        self._sessions[sid] = s
        return s

    def get(self, sid: str | None) -> Session | None:
        return self._sessions.get(sid) if sid else None

    def get_or_create(self, sid: str | None) -> Session:
        return self.get(sid) or self.create()
