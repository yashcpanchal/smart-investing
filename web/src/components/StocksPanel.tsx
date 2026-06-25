"use client";

import { useEffect, useRef, useState } from "react";
import { api, pct, type HoldingExplanation, type StockDetail } from "../lib/api";

export function StocksPanel({
  holdings,
  theme,
}: {
  holdings: HoldingExplanation[];
  theme: string;
}) {
  const [picked, setPicked] = useState<string | null>(null);
  const [details, setDetails] = useState<Record<string, StockDetail | "loading" | "error">>({});
  const requested = useRef<Set<string>>(new Set()); // keys we've already fetched (read only in effects)

  // derive the selected tab (defaults to the top holding) — no setState-in-effect
  const active = picked && holdings.some((h) => h.symbol === picked) ? picked : holdings[0]?.symbol ?? null;
  const setActive = setPicked;

  useEffect(() => {
    if (!active) return;
    const key = `${active}|${theme}`;
    if (requested.current.has(key)) return;
    requested.current.add(key);
    let cancelled = false;
    const sym = active;
    async function load() {
      setDetails((d) => ({ ...d, [sym]: "loading" }));
      try {
        const s = await api.stock(sym, theme);
        if (!cancelled) setDetails((d) => ({ ...d, [sym]: s }));
      } catch {
        if (!cancelled) setDetails((d) => ({ ...d, [sym]: "error" }));
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [active, theme]);

  if (!holdings.length) {
    return (
      <div className="grid h-full place-items-center text-center text-sm text-zinc-600">
        <p className="max-w-[240px]">Build a portfolio first — each holding gets a detail tab here.</p>
      </div>
    );
  }

  const sel = active ? details[active] : undefined;
  const hold = holdings.find((h) => h.symbol === active);

  return (
    <div className="flex h-full flex-col">
      {/* horizontal stock tabs */}
      <div className="flex gap-1.5 overflow-x-auto border-b border-zinc-800 px-3 py-2">
        {holdings.map((h) => (
          <button
            key={h.symbol}
            onClick={() => setActive(h.symbol)}
            className={`flex shrink-0 items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs transition ${
              active === h.symbol
                ? "border-cyan-400 bg-cyan-500/15 text-cyan-200"
                : "border-zinc-800 bg-zinc-900/40 text-zinc-300 hover:border-zinc-600"
            }`}
          >
            <span className="font-semibold">{h.symbol}</span>
            <span className="text-[10px] text-zinc-500">{pct(h.weight)}</span>
          </button>
        ))}
      </div>

      {/* detail dropdown for the selected stock */}
      <div className="flex-1 overflow-y-auto p-4">
        {sel === "loading" || sel === undefined ? (
          <div className="py-10 text-center text-sm text-zinc-500">
            <span className="dot">●</span> loading {active}…
          </div>
        ) : sel === "error" ? (
          <div className="text-sm text-red-400">Couldn&rsquo;t load {active}.</div>
        ) : (
          <div className="rise space-y-4">
            <div>
              <div className="flex items-center gap-2">
                <span className="grid h-9 min-w-[52px] place-items-center rounded-lg bg-zinc-800 px-2 text-sm font-semibold text-zinc-100">
                  {sel.symbol}
                </span>
                <div className="min-w-0">
                  <div className="truncate text-sm font-medium text-zinc-100">{sel.name}</div>
                  <div className="truncate text-[11px] text-zinc-500">
                    {[sel.sector, sel.industry].filter(Boolean).join(" · ") || "—"}
                  </div>
                </div>
                {hold && (
                  <span
                    className={`ml-auto rounded px-1.5 py-0.5 text-[10px] ${
                      hold.role === "direct" ? "bg-cyan-500/10 text-cyan-300" : "bg-violet-500/10 text-violet-300"
                    }`}
                  >
                    {hold.role}
                  </span>
                )}
              </div>
            </div>

            <Section label="How it relates to your thesis">
              <p className="text-sm leading-relaxed text-zinc-300">{sel.theme_fit}</p>
            </Section>

            {hold && (
              <Section label="Why it's in the portfolio">
                <p className="text-sm leading-relaxed text-zinc-300">
                  Holds <span className="font-semibold text-cyan-300">{pct(hold.weight)}</span> of the book as a{" "}
                  {hold.role === "direct" ? "direct thematic position" : "supply-chain position"}.{" "}
                  {hold.why}
                </p>
              </Section>
            )}

            <Section label="Key facts">
              <div className="grid grid-cols-3 gap-2">
                {sel.metrics.map((m) => (
                  <div key={m.label} className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-2">
                    <div className="text-[10px] text-zinc-500">{m.label}</div>
                    <div className="mt-0.5 text-sm font-semibold tabular-nums text-zinc-100">{m.value}</div>
                  </div>
                ))}
              </div>
            </Section>

            {sel.must_knows.length > 0 && (
              <Section label="Must-knows">
                <ul className="space-y-1.5">
                  {sel.must_knows.map((b, i) => (
                    <li key={i} className="flex gap-2 text-sm leading-relaxed text-zinc-300">
                      <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-cyan-400" />
                      {b}
                    </li>
                  ))}
                </ul>
              </Section>
            )}

            {sel.summary && (
              <Section label="Business">
                <p className="text-[13px] leading-relaxed text-zinc-400">{sel.summary}</p>
              </Section>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-zinc-500">{label}</div>
      {children}
    </div>
  );
}
