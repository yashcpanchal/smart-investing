export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export interface FrontierPoint {
  volatility: number;
  expected_return: number;
  sharpe: number;
}

export interface UniverseAsset {
  symbol: string;
  name: string;
  degree: number;
  rationale: string;
  scores: Record<string, number>;
}

export interface Trade {
  symbol: string;
  side: string;
  quantity: number;
  est_price: number | null;
}

export interface Explanation {
  summary: string;
  understood: string;
  selection: string;
  construction: string;
  risk_note: string;
  data_note: string;
  highlights: string[];
}

export interface Violation {
  code: string;
  message: string;
  severity: string;
}

export interface Proposal {
  id: string;
  spec: {
    raw_prompt: string;
    themes: string[];
    objective: string;
    include_indirect: boolean;
    exclude_symbols: string[];
    risk: { concentration_cap: number; target_volatility: number | null };
  };
  universe: { theme: string; assets: UniverseAsset[] };
  optimization: {
    weights: Record<string, number>;
    expected_return: number;
    volatility: number;
    sharpe: number;
    frontier: FrontierPoint[];
  };
  target_weights: Record<string, number>;
  trades: Trade[];
  backtest: {
    total_return: number;
    cagr: number;
    volatility: number;
    sharpe: number;
    max_drawdown: number;
  } | null;
  rationale: string;
  explanation: Explanation | null;
  blocked: boolean;
  violations: Violation[];
  lookback: string;
  price_source: string;
}

export interface ClarifyOption {
  label: string;
  value: string;
  hint?: string;
}
export interface ClarifyQuestion {
  id: string;
  question: string;
  help: string;
  kind: "single" | "multi";
  default?: string;
  options: ClarifyOption[];
}
export interface ClarifyResponse {
  interpretation: string;
  focus: string;
  questions: ClarifyQuestion[];
}

export interface ApproveResult {
  proposal_id: string;
  filled: number;
  rejected: number;
  realized_pnl: number;
  account: { cash: number; positions: Record<string, { symbol: string; quantity: number; avg_cost: number }> };
}

export interface CompileOpts {
  initial_cash?: number;
  answers?: Record<string, string | string[]>;
  lookback?: string;
}

async function jpost<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  return r.json();
}

async function jget<T>(path: string): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`);
  if (!r.ok) throw new Error(`${r.status}`);
  return r.json();
}

export const api = {
  clarify: (prompt: string) => jpost<ClarifyResponse>("/api/clarify", { prompt }),
  compile: (prompt: string, opts: CompileOpts = {}) =>
    jpost<Proposal>("/api/compile", { prompt, live: true, ...opts }),
  approve: (id: string) => jpost<ApproveResult>(`/api/proposals/${id}/approve`),
  portfolio: () => jget<unknown>("/api/portfolio"),
  health: () => jget<{ status: string; corpus_docs: number }>("/health"),
};

export const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
export const money = (x: number) =>
  x.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
export const money2 = (x: number) =>
  x.toLocaleString("en-US", { style: "currency", currency: "USD" });
