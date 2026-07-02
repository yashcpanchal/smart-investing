"use client";

import { useState } from "react";
import type { Explanation } from "../lib/api";

const SECTIONS: { key: keyof Explanation; label: string; icon: string }[] = [
  { key: "understood", label: "What I understood", icon: "◆" },
  { key: "selection", label: "Why these names", icon: "⌖" },
  { key: "construction", label: "How I built it", icon: "∿" },
  { key: "risk_note", label: "Risk & guardrails", icon: "⛨" },
  { key: "data_note", label: "Data window", icon: "▦" },
];

export function Findings({ explanation }: { explanation: Explanation }) {
  const [open, setOpen] = useState<string | null>("understood");

  return (
    <div className="rounded-2xl border border-cyan-500/20 bg-gradient-to-b from-cyan-500/[0.06] to-transparent p-5">
      <div className="flex items-start gap-3">
        <span className="mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-full bg-cyan-500/15 text-cyan-300">✦</span>
        <div>
          <h2 className="text-xs font-medium uppercase tracking-wide text-cyan-400/80">Here&rsquo;s what I found</h2>
          <p className="mt-1 text-[15px] leading-relaxed text-zinc-100">{explanation.summary}</p>
        </div>
      </div>

      {explanation.highlights.length > 0 && (
        <div className="mt-4 grid grid-cols-2 gap-2 lg:grid-cols-4">
          {explanation.highlights.map((h) => (
            <div key={h} className="rounded-xl border border-zinc-800 bg-zinc-900/40 px-3 py-2 text-xs text-zinc-300">
              {h}
            </div>
          ))}
        </div>
      )}

      <div className="mt-4 divide-y divide-zinc-800/70 rounded-xl border border-zinc-800/70 bg-zinc-950/30">
        {SECTIONS.map(({ key, label, icon }) => {
          const text = explanation[key] as string;
          if (!text) return null;
          const isOpen = open === key;
          return (
            <div key={key}>
              <button
                onClick={() => setOpen(isOpen ? null : key)}
                className="flex w-full items-center gap-3 px-4 py-3 text-left text-sm text-zinc-200 transition hover:bg-zinc-900/40"
              >
                <span className="text-zinc-500">{icon}</span>
                <span className="font-medium">{label}</span>
                <span className={`ml-auto text-zinc-500 transition-transform ${isOpen ? "rotate-90" : ""}`}>›</span>
              </button>
              {isOpen && (
                <p className="rise px-4 pb-4 pl-11 text-[13px] leading-relaxed text-zinc-400">{text}</p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
