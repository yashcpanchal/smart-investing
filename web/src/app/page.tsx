"use client";

import { useState } from "react";
import { AllocationChart } from "../components/AllocationChart";
import { FrontierChart } from "../components/FrontierChart";
import { api, money, pct, type ApproveResult, type Proposal } from "../lib/api";

const EXAMPLES = [
  "Invest in nuclear energy and uranium mining, include the supply chain, keep it lower risk and diversified",
  "Quantum computing and the companies that supply its cooling hardware",
  "Defense and aerospace primes, follow where the smart money is going",
];

export default function Home() {
  const [prompt, setPrompt] = useState(EXAMPLES[0]);
  const [cash, setCash] = useState(10000);
  const [topK, setTopK] = useState(12);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [proposal, setProposal] = useState<Proposal | null>(null);
  const [approving, setApproving] = useState(false);
  const [approved, setApproved] = useState<ApproveResult | null>(null);

  async function onCompile() {
    setLoading(true);
    setError(null);
    setApproved(null);
    setProposal(null);
    try {
      setProposal(await api.compile(prompt, { initial_cash: cash, top_k: topK }));
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function onApprove() {
    if (!proposal) return;
    setApproving(true);
    setError(null);
    try {
      setApproved(await api.approve(proposal.id));
    } catch (e) {
      setError(String(e));
    } finally {
      setApproving(false);
    }
  }

  const o = proposal?.optimization;

  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100">
      <div className="mx-auto max-w-5xl px-6 py-10">
        <header className="mb-8">
          <h1 className="text-3xl font-semibold tracking-tight">
            smart<span className="text-cyan-400">·</span>investing
          </h1>
          <p className="mt-1 text-sm text-zinc-400">
            Type a thesis. We search real filings, build a supply-chain-aware universe,
            optimize the weights with real math, and check every order — paper money.
          </p>
        </header>

        <section className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-5">
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={3}
            className="w-full resize-none rounded-xl border border-zinc-700 bg-zinc-950 p-3 text-sm outline-none focus:border-cyan-500"
            placeholder="e.g. Invest in nuclear and quantum, follow where private money is going…"
          />
          <div className="mt-2 flex flex-wrap gap-2">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                onClick={() => setPrompt(ex)}
                className="rounded-full border border-zinc-700 px-3 py-1 text-xs text-zinc-400 hover:border-cyan-600 hover:text-cyan-300"
              >
                {ex.slice(0, 38)}…
              </button>
            ))}
          </div>
          <div className="mt-4 flex flex-wrap items-end gap-5">
            <label className="text-xs text-zinc-400">
              Starting cash
              <input
                type="number"
                value={cash}
                onChange={(e) => setCash(+e.target.value)}
                className="mt-1 block w-32 rounded-lg border border-zinc-700 bg-zinc-950 px-2 py-1 text-sm text-zinc-100"
              />
            </label>
            <label className="text-xs text-zinc-400">
              Max holdings: <span className="text-zinc-200">{topK}</span>
              <input
                type="range"
                min={4}
                max={20}
                value={topK}
                onChange={(e) => setTopK(+e.target.value)}
                className="mt-2 block w-40 accent-cyan-500"
              />
            </label>
            <button
              onClick={onCompile}
              disabled={loading || !prompt.trim()}
              className="ml-auto rounded-xl bg-cyan-500 px-5 py-2 text-sm font-medium text-zinc-950 hover:bg-cyan-400 disabled:opacity-50"
            >
              {loading ? "Building…" : "Build portfolio"}
            </button>
          </div>
        </section>

        {error && (
          <div className="mt-4 rounded-xl border border-red-900 bg-red-950/40 p-3 text-sm text-red-300">
            {error}
            <div className="mt-1 text-xs text-red-400/70">
              Is the API running? Start it with <code>si serve</code> (and <code>si ingest …</code> for a corpus).
            </div>
          </div>
        )}

        {proposal && o && (
          <>
            <section className="mt-6 rounded-2xl border border-zinc-800 bg-zinc-900/50 p-5">
              <h2 className="text-sm font-medium text-zinc-300">What we understood</h2>
              <div className="mt-3 flex flex-wrap gap-2 text-xs">
                {proposal.spec.themes.map((t) => (
                  <span key={t} className="rounded-full bg-cyan-500/10 px-3 py-1 text-cyan-300">{t}</span>
                ))}
                <span className="rounded-full bg-zinc-800 px-3 py-1 text-zinc-300">objective: {proposal.spec.objective}</span>
                <span className="rounded-full bg-zinc-800 px-3 py-1 text-zinc-300">max/holding: {pct(proposal.spec.risk.concentration_cap)}</span>
                {proposal.spec.risk.target_volatility != null && (
                  <span className="rounded-full bg-zinc-800 px-3 py-1 text-zinc-300">target vol: {pct(proposal.spec.risk.target_volatility)}</span>
                )}
                <span className="rounded-full bg-zinc-800 px-3 py-1 text-zinc-300">supply chain: {proposal.spec.include_indirect ? "on" : "off"}</span>
                {proposal.spec.exclude_symbols.map((s) => (
                  <span key={s} className="rounded-full bg-red-500/10 px-3 py-1 text-red-300">excl {s}</span>
                ))}
              </div>
            </section>

            <section className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Metric label="Expected return" value={pct(o.expected_return)} />
              <Metric label="Volatility" value={pct(o.volatility)} />
              <Metric label="Sharpe" value={o.sharpe.toFixed(2)} highlight />
              {proposal.backtest && <Metric label="Backtest CAGR" value={pct(proposal.backtest.cagr)} />}
            </section>

            <section className="mt-6 grid gap-4 lg:grid-cols-2">
              <Card title="Efficient frontier">
                <FrontierChart frontier={o.frontier} chosen={{ volatility: o.volatility, expected_return: o.expected_return }} />
                <p className="mt-1 text-xs text-zinc-500">Cyan = achievable portfolios · amber = your optimized point</p>
              </Card>
              <Card title="Target allocation">
                <AllocationChart weights={proposal.target_weights} />
              </Card>
            </section>

            <Card title={`Asset universe — ${proposal.universe.assets.length} names`} className="mt-6">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="text-left text-xs uppercase text-zinc-500">
                    <tr>
                      <th className="py-2">Ticker</th>
                      <th>Name</th>
                      <th>Type</th>
                      <th className="text-right">Relevance</th>
                      <th className="text-right">Weight</th>
                    </tr>
                  </thead>
                  <tbody>
                    {proposal.universe.assets.map((a) => {
                      const w = proposal.target_weights[a.symbol] ?? 0;
                      const rel = a.scores.relevance ?? a.scores.graph_proximity ?? 0;
                      return (
                        <tr key={a.symbol} className="border-t border-zinc-800">
                          <td className="py-2 font-medium text-zinc-100">{a.symbol}</td>
                          <td className="max-w-[16rem] truncate text-zinc-400">{a.name}</td>
                          <td>
                            <span className={`rounded px-1.5 py-0.5 text-[10px] ${a.degree === 1 ? "bg-cyan-500/10 text-cyan-300" : "bg-violet-500/10 text-violet-300"}`}>
                              {a.degree === 1 ? "direct" : "supply-chain"}
                            </span>
                          </td>
                          <td className="text-right tabular-nums text-zinc-400">{rel.toFixed(2)}</td>
                          <td className="text-right tabular-nums text-zinc-100">{pct(w)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Card>

            <section className="mt-6 rounded-2xl border border-zinc-800 bg-zinc-900/50 p-5">
              <p className="text-sm text-zinc-300">{proposal.rationale}</p>
              <div className="mt-4 flex items-center gap-3">
                <button
                  onClick={onApprove}
                  disabled={approving || proposal.trades.length === 0}
                  className="rounded-xl bg-emerald-500 px-5 py-2 text-sm font-medium text-zinc-950 hover:bg-emerald-400 disabled:opacity-50"
                >
                  {approving ? "Executing…" : `Approve & execute ${proposal.trades.length} orders (paper)`}
                </button>
                {proposal.trades.length === 0 && (
                  <span className="text-xs text-amber-400">Blocked by circuit breaker — nothing to execute.</span>
                )}
              </div>
              {approved && (
                <div className="mt-4 rounded-xl border border-emerald-900 bg-emerald-950/30 p-3 text-sm text-emerald-200">
                  Executed {approved.filled} orders ({approved.rejected} rejected). Cash left{" "}
                  {money(approved.account.cash)} · realized P&amp;L {money(approved.realized_pnl)}.
                  <div className="mt-1 text-xs text-emerald-300/70">
                    Holdings: {Object.values(approved.account.positions).map((p) => `${p.symbol} ${p.quantity.toFixed(2)}`).join(" · ")}
                  </div>
                </div>
              )}
            </section>
          </>
        )}

        <footer className="mt-10 text-center text-xs text-zinc-600">
          Paper trading. Not investment advice. Robinhood Agentic MCP execution comes later.
        </footer>
      </div>
    </main>
  );
}

function Metric({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
      <div className="text-xs text-zinc-500">{label}</div>
      <div className={`mt-1 text-xl font-semibold tabular-nums ${highlight ? "text-cyan-400" : "text-zinc-100"}`}>{value}</div>
    </div>
  );
}

function Card({ title, children, className = "" }: { title: string; children: React.ReactNode; className?: string }) {
  return (
    <div className={`rounded-2xl border border-zinc-800 bg-zinc-900/50 p-5 ${className}`}>
      <h2 className="mb-3 text-sm font-medium text-zinc-300">{title}</h2>
      {children}
    </div>
  );
}
