"""Shared domain contracts. Every later phase imports from here.

These are the nouns of the whole system: a StrategySpec (what the user wants) is
compiled into an AssetUniverse (what to buy from), optimized into target weights,
turned into Orders, validated by the circuit breaker, and executed against an
AccountState by a Broker.

Money note: amounts are plain floats for MVP. The broker rounds to cents to avoid
drift. A Decimal/cents-int migration is deferred until Phase 11 (Robinhood
reconciliation), where exact equality matters.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    REJECTED = "rejected"
    CANCELED = "canceled"


class Objective(str, Enum):
    MAX_SHARPE = "max_sharpe"
    MIN_VOL = "min_vol"
    TARGET_VOL = "target_vol"


class Severity(str, Enum):
    FATAL = "fatal"  # blocks execution
    WARN = "warn"  # informational


# --------------------------------------------------------------------------- #
# Orders & account state
# --------------------------------------------------------------------------- #
class Order(BaseModel):
    symbol: str
    side: OrderSide
    quantity: float = Field(gt=0, description="Shares; fractional allowed.")
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    est_price: float | None = Field(default=None, description="Price estimate at proposal time.")
    id: str = Field(default_factory=_new_id, description="Client-side order id (idempotency key).")
    created_at: str = Field(default_factory=_utcnow_iso)

    @property
    def est_notional(self) -> float | None:
        if self.est_price is None:
            return None
        return self.quantity * self.est_price


class Position(BaseModel):
    symbol: str
    quantity: float
    avg_cost: float = Field(description="Weighted-average cost basis per share.")

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_cost


class AccountState(BaseModel):
    cash: float
    positions: dict[str, Position] = Field(default_factory=dict)
    # Robinhood exposes buying_power directly; for cash accounts it == cash.
    buying_power: float | None = None
    # Round-trip day trades in the rolling 5 business days (for PDT rule).
    day_trades_5d: int = 0
    as_of: str = Field(default_factory=_utcnow_iso)

    def effective_buying_power(self) -> float:
        return self.buying_power if self.buying_power is not None else self.cash

    def market_value(self, prices: dict[str, float]) -> float:
        return sum(p.quantity * prices.get(s, p.avg_cost) for s, p in self.positions.items())

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + self.market_value(prices)

    def weights(self, prices: dict[str, float]) -> dict[str, float]:
        eq = self.equity(prices)
        if eq <= 0:
            return {}
        return {s: (p.quantity * prices.get(s, p.avg_cost)) / eq for s, p in self.positions.items()}


class OrderResult(BaseModel):
    order: Order
    status: OrderStatus
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    realized_pnl: float = 0.0
    broker_order_id: str | None = None
    filled_at: str | None = None
    message: str = ""


# --------------------------------------------------------------------------- #
# Strategy specification (the compiled user intent)
# --------------------------------------------------------------------------- #
class RiskParams(BaseModel):
    concentration_cap: float = Field(default=0.30, description="Max weight per single asset.")
    target_volatility: float | None = Field(default=None, description="Annualized, for target_vol objective.")
    max_positions: int = 25
    allow_short: bool = False
    cash_buffer: float = Field(default=0.0, description="Fraction of equity kept uninvested.")


class SourceWeights(BaseModel):
    """Relative weighting of evidence sources in universe ranking (the UI sliders)."""

    sec_13f: float = 0.5
    insider: float = 0.2
    news_sentiment: float = 0.2
    social: float = 0.1


class RebalanceConfig(BaseModel):
    cadence_days: int = 14
    autonomous: bool = Field(default=False, description="Execute without per-trade approval if True.")


class StrategySpec(BaseModel):
    raw_prompt: str = ""
    themes: list[str] = Field(default_factory=list)
    include_symbols: list[str] = Field(default_factory=list)
    exclude_symbols: list[str] = Field(default_factory=list)
    include_indirect: bool = Field(default=True, description="Pull 2nd-degree supply-chain names.")
    max_indirect_hops: int = 1
    source_weights: SourceWeights = Field(default_factory=SourceWeights)
    risk: RiskParams = Field(default_factory=RiskParams)
    rebalance: RebalanceConfig = Field(default_factory=RebalanceConfig)
    objective: Objective = Objective.MAX_SHARPE


# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #
class UniverseAsset(BaseModel):
    symbol: str
    name: str = ""
    degree: int = Field(default=1, description="1 = direct match, 2 = indirect / supply-chain.")
    rationale: str = ""
    scores: dict[str, float] = Field(default_factory=dict)


class AssetUniverse(BaseModel):
    theme: str = ""
    assets: list[UniverseAsset] = Field(default_factory=list)

    @property
    def symbols(self) -> list[str]:
        return [a.symbol for a in self.assets]


# --------------------------------------------------------------------------- #
# Optimization & backtest outputs
# --------------------------------------------------------------------------- #
class FrontierPoint(BaseModel):
    volatility: float
    expected_return: float
    sharpe: float
    weights: dict[str, float] = Field(default_factory=dict)


class OptimizationResult(BaseModel):
    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe: float
    objective: Objective = Objective.MAX_SHARPE
    frontier: list[FrontierPoint] = Field(default_factory=list)


class BacktestResult(BaseModel):
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    max_drawdown: float
    start: str = ""
    end: str = ""
    n_days: int = 0


# --------------------------------------------------------------------------- #
# Validation (circuit breaker)
# --------------------------------------------------------------------------- #
class Violation(BaseModel):
    code: str
    message: str
    severity: Severity = Severity.FATAL


class ValidationResult(BaseModel):
    ok: bool
    violations: list[Violation] = Field(default_factory=list)

    @property
    def fatal(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == Severity.FATAL]

    @property
    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == Severity.WARN]


# --------------------------------------------------------------------------- #
# Explanation (plain-English "explain our findings", grounded in real numbers)
# --------------------------------------------------------------------------- #
class HoldingExplanation(BaseModel):
    """Per-name reasoning, grounded in the real pipeline numbers."""

    symbol: str
    name: str = ""
    weight: float = 0.0
    role: str = "direct"  # "direct" | "supply-chain"
    relevance: float = 0.0  # cosine to the theme (direct) or graph proximity (indirect)
    why: str = ""  # one-line, grounded rationale


class Explanation(BaseModel):
    """Human-readable account of what we did and why. Every numeric claim is
    filled from the actual pipeline outputs (never the LLM) so it can't drift;
    the LLM only polishes the prose."""

    summary: str = ""  # 1-2 sentence plain-English headline
    understood: str = ""  # what we read from the thesis
    selection: str = ""  # why these names (direct vs supply-chain)
    construction: str = ""  # how the weights were chosen (the math, in words)
    risk_note: str = ""  # guardrails / circuit breaker / concentration
    data_note: str = ""  # data window + source
    highlights: list[str] = Field(default_factory=list)  # quick key-findings bullets
    holdings: list[HoldingExplanation] = Field(default_factory=list)  # per-name reasoning


# --------------------------------------------------------------------------- #
# Proposal (what we show the user before executing)
# --------------------------------------------------------------------------- #
class Proposal(BaseModel):
    id: str = Field(default_factory=_new_id)
    spec: StrategySpec
    universe: AssetUniverse
    optimization: OptimizationResult
    target_weights: dict[str, float]
    trades: list[Order] = Field(default_factory=list)
    backtest: BacktestResult | None = None
    rationale: str = ""
    explanation: Explanation | None = None
    blocked: bool = False  # circuit breaker blocked execution
    violations: list[Violation] = Field(default_factory=list)
    lookback: str = "2y"  # analysis window used for prices/backtest
    price_source: str = ""  # "yfinance (live)" | "synthetic (offline)"
    generated_at: str = Field(default_factory=_utcnow_iso)
