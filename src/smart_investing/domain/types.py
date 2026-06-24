"""Shared domain contracts. Every later phase imports from here.

These are the nouns of the whole system: a StrategySpec (what the user wants) is
compiled into an AssetUniverse (what to buy from), optimized into TargetWeights,
turned into Orders, validated, and executed against an AccountState by a Broker.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Orders & account state
# --------------------------------------------------------------------------- #
class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class Order(BaseModel):
    symbol: str
    side: OrderSide
    quantity: float = Field(gt=0, description="Shares; fractional allowed.")
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    est_price: float | None = Field(default=None, description="Price estimate at proposal time.")

    @property
    def est_notional(self) -> float | None:
        if self.est_price is None:
            return None
        return self.quantity * self.est_price


class Position(BaseModel):
    symbol: str
    quantity: float
    avg_cost: float = Field(description="Cost basis per share.")

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_cost


class AccountState(BaseModel):
    cash: float
    positions: dict[str, Position] = Field(default_factory=dict)
    # Round-trip day trades in the rolling 5 business days (for PDT rule).
    day_trades_5d: int = 0

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
    status: str  # "filled" | "rejected" | "partial"
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    realized_pnl: float = 0.0
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
    objective: str = Field(default="max_sharpe", description="max_sharpe | min_vol | target_vol")


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
    objective: str = "max_sharpe"
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
    severity: str = "fatal"  # "fatal" blocks execution; "warn" is informational


class ValidationResult(BaseModel):
    ok: bool
    violations: list[Violation] = Field(default_factory=list)

    @property
    def fatal(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "fatal"]


# --------------------------------------------------------------------------- #
# Proposal (what we show the user before executing)
# --------------------------------------------------------------------------- #
class Proposal(BaseModel):
    spec: StrategySpec
    universe: AssetUniverse
    optimization: OptimizationResult
    target_weights: dict[str, float]
    trades: list[Order] = Field(default_factory=list)
    backtest: BacktestResult | None = None
    rationale: str = ""
    generated_at: str = Field(default_factory=_utcnow_iso)
