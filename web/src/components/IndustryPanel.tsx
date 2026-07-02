"use client";

import { useEffect, useState } from "react";
import { api, type IndustryBrief } from "../lib/api";

export function IndustryPanel({ theme }: { theme: string }) {
  const [brief, setBrief] = useState<IndustryBrief | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!theme) return;
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const b = await api.industry(theme);
        if (!cancelled) setBrief(b);
      } catch (e) {
        if (!cancelled) setError(String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [theme]);

  if (!theme) {
    return (
      <div className="grid h-full place-items-center text-center text-sm text-zinc-600">
        <p className="max-w-[240px]">Describe a theme to see the industry map and current market state.</p>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto p-4">
      {loading && !brief ? (
        <div className="py-10 text-center text-sm text-zinc-500">
          <span className="dot">●</span> researching the industry…
        </div>
      ) : error ? (
        <div className="text-sm text-red-400">{error}</div>
      ) : brief ? (
        <div className="rise space-y-5">
          <div>
            <div className="text-[11px] uppercase tracking-wide text-cyan-400/70">Industry map</div>
            <h2 className="text-lg font-semibold capitalize text-zinc-50">{brief.theme}</h2>
          </div>

          {/* supply-chain layers */}
          <div className="space-y-2">
            {brief.supply_chain.map((layer, i) => (
              <div key={layer.layer}>
                <div className="mb-1 text-[11px] font-medium text-zinc-400">{layer.layer}</div>
                <div className="flex flex-wrap gap-1.5">
                  {layer.players.map((p) => (
                    <span
                      key={p.symbol}
                      title={p.name}
                      className="rounded-lg border border-zinc-700 bg-zinc-900/60 px-2 py-1 text-xs text-zinc-200"
                    >
                      <span className="font-semibold">{p.symbol}</span>
                    </span>
                  ))}
                </div>
                {i < brief.supply_chain.length - 1 && <div className="my-1 pl-1 text-zinc-600">↓</div>}
              </div>
            ))}
          </div>

          {/* current state */}
          <div className="rounded-2xl border border-cyan-500/20 bg-gradient-to-b from-cyan-500/[0.06] to-transparent p-4">
            <div className="flex items-center gap-2">
              <h3 className="text-xs font-medium uppercase tracking-wide text-cyan-400/80">State of the market</h3>
              <span
                className={`rounded px-1.5 py-0.5 text-[9px] ${
                  brief.grounded ? "bg-emerald-500/15 text-emerald-300" : "bg-zinc-700/50 text-zinc-400"
                }`}
              >
                {brief.grounded ? "live · grounded" : "offline"}
              </span>
            </div>
            <p className="mt-2 text-sm leading-relaxed text-zinc-200">{brief.analysis}</p>
          </div>

          {(brief.tailwinds.length > 0 || brief.risks.length > 0) && (
            <div className="grid gap-3 sm:grid-cols-2">
              {brief.tailwinds.length > 0 && (
                <Bullets label="Tailwinds" color="emerald" items={brief.tailwinds} />
              )}
              {brief.risks.length > 0 && <Bullets label="Risks" color="rose" items={brief.risks} />}
            </div>
          )}

          {brief.outlook && (
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-3">
              <div className="text-[11px] font-medium uppercase tracking-wide text-zinc-500">Outlook</div>
              <p className="mt-1 text-sm leading-relaxed text-zinc-300">{brief.outlook}</p>
            </div>
          )}

          {brief.sources.length > 0 && (
            <div>
              <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-zinc-500">Sources</div>
              <div className="flex flex-col gap-1">
                {brief.sources.slice(0, 6).map((s, i) => (
                  <a
                    key={i}
                    href={s.uri}
                    target="_blank"
                    rel="noreferrer"
                    className="truncate text-xs text-cyan-400/80 hover:text-cyan-300 hover:underline"
                  >
                    ↗ {s.title || s.uri}
                  </a>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}

function Bullets({ label, color, items }: { label: string; color: "emerald" | "rose"; items: string[] }) {
  const dot = color === "emerald" ? "bg-emerald-400" : "bg-rose-400";
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-3">
      <div className={`text-[11px] font-medium uppercase tracking-wide ${color === "emerald" ? "text-emerald-400/80" : "text-rose-400/80"}`}>
        {label}
      </div>
      <ul className="mt-1.5 space-y-1.5">
        {items.map((b, i) => (
          <li key={i} className="flex gap-2 text-[13px] leading-relaxed text-zinc-300">
            <span className={`mt-1.5 h-1 w-1 shrink-0 rounded-full ${dot}`} />
            {b}
          </li>
        ))}
      </ul>
    </div>
  );
}
