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

export interface HoldingExplanation {
  symbol: string;
  name: string;
  weight: number;
  role: string; // "direct" | "supply-chain"
  relevance: number;
  why: string;
}

export interface Explanation {
  summary: string;
  understood: string;
  selection: string;
  construction: string;
  risk_note: string;
  data_note: string;
  highlights: string[];
  holdings: HoldingExplanation[];
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

// ---- supply-chain graph ----
export interface GraphNode {
  symbol: string;
  name: string;
  tradeable: boolean;
  relevance: number;
}
export interface GraphNeighbor extends GraphNode {
  direction: "upstream" | "downstream" | "peer" | "related";
  rel: string; // "supplies" | "competes" | "co_mention"
  weight: number;
  origin: string;
}
export interface SearchResponse {
  theme: string;
  nodes: GraphNode[];
}
export interface NeighborsResponse {
  node: GraphNode;
  neighbors: GraphNeighbor[];
}

// ---- conversation ----
export interface ChatState {
  id: string;
  theme: string;
  search_theme: string;
  risk: string;
  breadth: string;
  supply_chain: string;
  lookback: string;
  cash: number;
  pinned: string[];
  excluded: string[];
}
export interface ChatAction {
  op: string;
  [k: string]: unknown;
}
export interface ChatResponse {
  session_id: string;
  reply: string;
  actions: ChatAction[];
  researched?: string[]; // read tools the agent used before replying
  added: string[];
  removed: string[];
  rebuilt: boolean;
  proposal: Proposal | null;
  state: ChatState;
}

// ---- per-stock detail ----
export interface StockMetric {
  label: string;
  value: string;
}
export interface StockDetail {
  symbol: string;
  name: string;
  sector: string;
  industry: string;
  summary: string;
  metrics: StockMetric[];
  theme_fit: string;
  must_knows: string[];
}

// ---- industry briefing ----
export interface SupplyLayer {
  layer: string;
  players: { symbol: string; name: string }[];
}
export interface IndustryBrief {
  theme: string;
  supply_chain: SupplyLayer[];
  analysis: string;
  tailwinds: string[];
  risks: string[];
  outlook: string;
  sources: { title: string; uri: string }[];
  grounded: boolean;
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
  chat: (message: string, session_id?: string | null) =>
    jpost<ChatResponse>("/api/chat", { message, session_id: session_id ?? null, live: true }),
  graphSearch: (theme: string, top_k = 8) =>
    jget<SearchResponse>(`/api/graph/search?theme=${encodeURIComponent(theme)}&top_k=${top_k}`),
  graphNeighbors: (node: string, theme = "", limit = 12) =>
    jget<NeighborsResponse>(
      `/api/graph/neighbors?node=${encodeURIComponent(node)}&theme=${encodeURIComponent(theme)}&limit=${limit}`,
    ),
  stock: (symbol: string, theme = "") =>
    jget<StockDetail>(`/api/stock/${encodeURIComponent(symbol)}?theme=${encodeURIComponent(theme)}`),
  industry: (theme: string) => jget<IndustryBrief>(`/api/industry?theme=${encodeURIComponent(theme)}`),
  approve: (id: string) => jpost<ApproveResult>(`/api/proposals/${id}/approve`),
  portfolio: () => jget<unknown>("/api/portfolio"),
  health: () => jget<{ status: string; corpus_docs: number }>("/health"),
};

export const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
export const money = (x: number) =>
  x.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
export const money2 = (x: number) =>
  x.toLocaleString("en-US", { style: "currency", currency: "USD" });
