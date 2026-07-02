"use client";

import { api, money2, pct, type ApproveResult, type ChatState, type Proposal } from "../lib/api";
import { AllocationChart } from "./AllocationChart";
import { Findings } from "./Findings";
import { FrontierChart } from "./FrontierChart";
import { useState } from "react";

export function PortfolioPanel({
  proposal,
  busy,
  onRemove,
  sessionId,
  chatState,
  onKnobs,
}: {
  proposal: Proposal | null;
  busy: boolean;
  onRemove: (symbol: string) => void;
  sessionId?: string | null;
  chatState?: ChatState | null;
  onKnobs?: (proposal: Proposal | null, state: ChatState) => void;
}) {
  const [approving, setApproving] = useState(false);
  const [approved, setApproved] = useState<ApproveResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function approve() {
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

  if (!proposal) {
    return (
      <div className="grid h-full place-items-center text-center text-sm text-zinc-600">
        <div>
          <div className="text-3xl">◷</div>
          <p className="mt-2 max-w-[220px]">Your portfolio and the reasoning behind it will build here.</p>
        </div>
      </div>
    );
  }

  const o = proposal.optimization;
  const exp = proposal.explanation;
  const held = exp?.holdings ?? [];

  return (
    <div className="h-full space-y-5 overflow-y-auto px-4 py-4">
      <div className="flex items-end justify-between gap-2">
        <div>
          <div className="text-[11px] uppercase tracking-wide text-cyan-400/70">Portfolio</div>
          <h2 className="text-lg font-semibold tracking-tight text-zinc-50">
            {proposal.spec.themes[0] || proposal.universe.theme || "Custom thesis"}
          </h2>
        </div>
        <span className="text-[10px] text-zinc-600">{proposal.price_source}</span>
      </div>

      {proposal.blocked && (
        <div className="rounded-xl border border-amber-800/60 bg-amber-950/20 p-3 text-xs text-amber-200">
          <div className="font-medium">Circuit breaker blocked execution</div>
          <ul className="mt-1 list-disc space-y-0.5 pl-4 text-amber-300/80">
            {proposal.violations.filter((v) => v.severity === "fatal").map((v) => (
              <li key={v.code}>{v.message}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="grid grid-cols-2 gap-2">
        <Metric label="Expected return" value={pct(o.expected_return)} />
        <Metric label="Risk (vol)" value={pct(o.volatility)} />
        <Metric label="Sharpe" value={o.sharpe.toFixed(2)} highlight />
        <Metric
          label={proposal.backtest ? `Backtest CAGR · ${proposal.lookback}` : "Holdings"}
          value={proposal.backtest ? pct(proposal.backtest.cagr) : String(held.length)}
        />
      </div>

      {sessionId && chatState && onKnobs && (
        <SignalWeights
          // remount when the session's weights change elsewhere (chat, reset)
          key={`${chatState.source_weights?.sec_13f}-${chatState.source_weights?.insider}`}
          sessionId={sessionId}
          chatState={chatState}
          busy={busy}
          onKnobs={onKnobs}
        />
      )}

      {exp && <Findings explanation={exp} />}

      {held.length > 0 && (
        <div>
          <div className="mb-2 text-xs font-medium uppercase tracking-wide text-zinc-500">
            Holdings — {held.length} names
          </div>
          <div className="space-y-1.5">
            {held.map((h) => (
              <div key={h.symbol} className="group rounded-xl border border-zinc-800 bg-zinc-900/40 p-2.5">
                <div className="flex items-center gap-2">
                  <span className="grid h-7 min-w-[44px] place-items-center rounded-lg bg-zinc-800/70 px-1 text-xs font-semibold text-zinc-100">
                    {h.symbol}
                  </span>
                  <span className="truncate text-xs text-zinc-300">{h.name || h.symbol}</span>
                  <span
                    className={`ml-auto rounded px-1.5 py-0.5 text-[9px] ${
                      h.role === "direct" ? "bg-cyan-500/10 text-cyan-300" : "bg-violet-500/10 text-violet-300"
                    }`}
                  >
                    {h.role}
                  </span>
                  {h.smart_money != null && (
                    <span
                      title={`Smart money score ${(h.smart_money * 100).toFixed(0)}% — institutions/insiders buying`}
                      className="rounded bg-emerald-500/10 px-1.5 py-0.5 text-[9px] text-emerald-300"
                    >
                      smart $
                    </span>
                  )}
                  <span className="w-12 text-right text-xs font-medium tabular-nums text-zinc-100">{pct(h.weight)}</span>
                  <button
                    onClick={() => onRemove(h.symbol)}
                    disabled={busy}
                    title={`Remove ${h.symbol}`}
                    className="grid h-5 w-5 place-items-center rounded border border-zinc-700 text-[10px] text-zinc-500 opacity-0 transition hover:border-red-700 hover:text-red-400 group-hover:opacity-100 disabled:opacity-30"
                  >
                    ×
                  </button>
                </div>
                <div className="mt-1.5 h-1 w-full overflow-hidden rounded-full bg-zinc-800">
                  <div className="h-full rounded-full bg-cyan-400" style={{ width: `${Math.min(h.weight * 100 * 2.2, 100)}%` }} />
                </div>
                {h.why && <p className="mt-1.5 text-[11px] leading-relaxed text-zinc-400">{h.why}</p>}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="grid gap-3">
        <Card title="Allocation">
          <AllocationChart weights={proposal.target_weights} />
        </Card>
        <Card title="Efficient frontier" subtitle="amber dot = your optimized portfolio">
          <FrontierChart frontier={o.frontier} chosen={{ volatility: o.volatility, expected_return: o.expected_return }} />
        </Card>
      </div>

      <div className="sticky bottom-0 -mx-4 border-t border-zinc-800 bg-zinc-950/90 px-4 py-3 backdrop-blur">
        {error && <div className="mb-2 text-xs text-red-400">{error}</div>}
        <button
          onClick={approve}
          disabled={approving || busy || proposal.trades.length === 0}
          className="w-full rounded-xl bg-emerald-500 px-4 py-2.5 text-sm font-semibold text-zinc-950 transition hover:bg-emerald-400 disabled:opacity-50"
        >
          {approving ? "Executing…" : `Approve & execute ${proposal.trades.length} orders (paper)`}
        </button>
        {approved && (
          <div className="rise mt-2 rounded-xl border border-emerald-900/60 bg-emerald-950/20 p-2.5 text-xs text-emerald-200">
            ✓ Executed {approved.filled} orders ({approved.rejected} rejected). Cash {money2(approved.account.cash)} ·
            realized P&amp;L {money2(approved.realized_pnl)}.
          </div>
        )}
      </div>
    </div>
  );
}

const DEFAULT_WEIGHTS = { sec_13f: 0.5, insider: 0.2 };

function SignalWeights({
  sessionId,
  chatState,
  busy,
  onKnobs,
}: {
  sessionId: string;
  chatState: ChatState;
  busy: boolean;
  onKnobs: (proposal: Proposal | null, state: ChatState) => void;
}) {
  const remote13f = chatState.source_weights?.sec_13f ?? DEFAULT_WEIGHTS.sec_13f;
  const remoteInsider = chatState.source_weights?.insider ?? DEFAULT_WEIGHTS.insider;
  // initial values only — the parent remounts this component (via key) when
  // the session's weights change through another path
  const [w13f, setW13f] = useState(remote13f);
  const [wInsider, setWInsider] = useState(remoteInsider);
  const [saving, setSaving] = useState(false);

  async function commit() {
    if (saving || (w13f === remote13f && wInsider === remoteInsider)) return;
    setSaving(true);
    try {
      const r = await api.setKnobs(sessionId, { sec_13f: w13f, insider: wInsider });
      onKnobs(r.proposal, r.state);
    } catch {
      // keep the local values; the next release retries
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-3">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-xs font-medium text-zinc-200">Signal weights</h3>
        <span className="text-[10px] text-zinc-500">{saving ? "reweighting…" : "smart money"}</span>
      </div>
      <Slider
        label="13F institutions"
        hint="quarterly holdings of Berkshire, RenTech & co"
        value={w13f}
        disabled={busy || saving}
        onChange={setW13f}
        onCommit={commit}
      />
      <Slider
        label="Insider buys"
        hint="recent open-market Form 4 purchases"
        value={wInsider}
        disabled={busy || saving}
        onChange={setWInsider}
        onCommit={commit}
      />
    </div>
  );
}

function Slider({
  label,
  hint,
  value,
  disabled,
  onChange,
  onCommit,
}: {
  label: string;
  hint: string;
  value: number;
  disabled: boolean;
  onChange: (v: number) => void;
  onCommit: () => void;
}) {
  return (
    <label className="mb-2 block last:mb-0">
      <div className="flex items-baseline justify-between">
        <span className="text-[11px] text-zinc-400" title={hint}>
          {label}
        </span>
        <span className="text-[11px] tabular-nums text-zinc-300">{value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min={0}
        max={1}
        step={0.05}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        onPointerUp={onCommit}
        onKeyUp={onCommit}
        className="mt-1 w-full accent-cyan-400"
      />
    </label>
  );
}

function Metric({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-2.5">
      <div className="text-[11px] text-zinc-500">{label}</div>
      <div className={`mt-0.5 text-lg font-semibold tabular-nums ${highlight ? "text-cyan-400" : "text-zinc-100"}`}>
        {value}
      </div>
    </div>
  );
}

function Card({ title, subtitle, children }: { title: string; subtitle?: string; children: React.ReactNode }) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-3">
      <div className="mb-2">
        <h3 className="text-xs font-medium text-zinc-200">{title}</h3>
        {subtitle && <p className="text-[10px] text-zinc-500">{subtitle}</p>}
      </div>
      {children}
    </div>
  );
}
