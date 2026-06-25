"use client";

import type { Proposal } from "../lib/api";
import { pct } from "../lib/api";

export function HoldingsTable({
  proposal,
  onPrune,
  pruning,
}: {
  proposal: Proposal;
  onPrune: (symbol: string) => void;
  pruning: string | null;
}) {
  const rows = proposal.universe.assets
    .map((a) => ({ a, w: proposal.target_weights[a.symbol] ?? 0 }))
    .sort((x, y) => y.w - x.w);
  const maxW = Math.max(...rows.map((r) => r.w), 0.0001);

  return (
    <div className="space-y-1.5">
      {rows.map(({ a, w }) => {
        const rel = a.scores.relevance ?? a.scores.graph_proximity ?? 0;
        const held = w > 0.005;
        return (
          <div
            key={a.symbol}
            className={`group grid grid-cols-[auto_1fr_auto] items-center gap-3 rounded-xl border px-3 py-2.5 transition ${
              held ? "border-zinc-800 bg-zinc-900/40" : "border-zinc-800/50 bg-zinc-900/20 opacity-60"
            }`}
          >
            <div className="flex items-center gap-2.5">
              <span className="grid h-9 w-12 place-items-center rounded-lg bg-zinc-800/70 text-xs font-semibold tracking-tight text-zinc-100">
                {a.symbol}
              </span>
              <span
                className={`hidden rounded px-1.5 py-0.5 text-[10px] sm:inline ${
                  a.degree === 1 ? "bg-cyan-500/10 text-cyan-300" : "bg-violet-500/10 text-violet-300"
                }`}
              >
                {a.degree === 1 ? "direct" : "supply-chain"}
              </span>
            </div>

            <div className="min-w-0">
              <div className="truncate text-sm text-zinc-200">{a.name || a.symbol}</div>
              <div className="mt-0.5 truncate text-[11px] text-zinc-500" title={a.rationale}>
                {a.rationale} · relevance {rel.toFixed(2)}
              </div>
              <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
                <div
                  className={`h-full rounded-full ${held ? "bg-cyan-400" : "bg-zinc-600"}`}
                  style={{ width: `${Math.max((w / maxW) * 100, held ? 6 : 0)}%` }}
                />
              </div>
            </div>

            <div className="flex items-center gap-2">
              <span className="w-14 text-right text-sm font-medium tabular-nums text-zinc-100">
                {held ? pct(w) : "—"}
              </span>
              <button
                onClick={() => onPrune(a.symbol)}
                disabled={pruning !== null}
                title={`Remove ${a.symbol} and rebuild`}
                className="grid h-7 w-7 place-items-center rounded-lg border border-zinc-700 text-zinc-500 opacity-0 transition hover:border-red-700 hover:text-red-400 group-hover:opacity-100 disabled:opacity-40"
              >
                {pruning === a.symbol ? <span className="dot">·</span> : "×"}
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}
