"use client";

import { useEffect, useRef, useState } from "react";
import type { ResearchEntry } from "../lib/api";

export interface Msg {
  role: "user" | "assistant";
  text: string;
  researched?: ResearchEntry[]; // read-tool trace the agent produced before replying
}

export interface ChatProgress {
  researched: ResearchEntry[]; // calls seen so far this turn (preview fills in as results land)
  status: string; // "thinking…" | "researching…" | "queuing changes…"
}

const TOOL_LABELS: Record<string, string> = {
  get_portfolio: "checked the portfolio",
  get_stock_facts: "looked up company facts",
  get_neighbors: "walked the supply chain",
  search_companies: "searched SEC filings",
  web_research: "searched the web",
};

/** The one argument worth showing next to the label (ticker or query). */
function keyArg(e: ResearchEntry): string {
  const v = e.args?.symbol ?? e.args?.query;
  return typeof v === "string" && v ? (v.length > 32 ? v.slice(0, 32) + "…" : v) : "";
}

/** Dedupe for display by (tool + args) while keeping first-seen order. */
function dedupe(entries: ResearchEntry[]): ResearchEntry[] {
  const seen = new Map<string, ResearchEntry>();
  for (const e of entries) {
    const k = `${e.tool}:${JSON.stringify(e.args ?? {})}`;
    if (!seen.has(k)) seen.set(k, e);
  }
  return [...seen.values()];
}

function ResearchPills({ entries }: { entries: ResearchEntry[] }) {
  if (entries.length === 0) return null;
  return (
    <div className="mb-1 flex flex-wrap gap-1 px-1">
      {dedupe(entries).map((e, i) => {
        const arg = keyArg(e);
        return (
          <span
            key={i}
            title={e.preview || undefined}
            className="cursor-default rounded-full border border-zinc-800 bg-zinc-900/60 px-2 py-0.5 text-[10px] text-zinc-500"
          >
            🔍 {TOOL_LABELS[e.tool] ?? e.tool}
            {arg && <span className="text-zinc-400"> · {arg}</span>}
          </span>
        );
      })}
    </div>
  );
}

const EXAMPLES = [
  "The AI data-center buildout: chips, power, and cooling",
  "Nuclear energy and the uranium supply chain, lower risk",
  "Defense and aerospace primes",
];

const HINTS = [
  '"go deeper on NVDA\'s suppliers"',
  '"make it safer"',
  '"drop the consumer names"',
  '"why is MU in here?"',
  '"spread it across more names"',
];

export function ChatPanel({
  messages,
  onSend,
  busy,
  started,
  progress = null,
}: {
  messages: Msg[];
  onSend: (text: string) => void;
  busy: boolean;
  started: boolean;
  progress?: ChatProgress | null;
}) {
  const [text, setText] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  function submit() {
    const t = text.trim();
    if (!t || busy) return;
    onSend(t);
    setText("");
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
        {!started && (
          <div className="rise">
            <h2 className="text-lg font-semibold text-zinc-100">What do you want to invest in?</h2>
            <p className="mt-1 text-sm leading-relaxed text-zinc-400">
              Describe a theme. I&rsquo;ll search real SEC filings, map the supply chain, and build an
              optimized portfolio — then we refine it together.
            </p>
            <div className="mt-4 space-y-2">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  onClick={() => onSend(ex)}
                  disabled={busy}
                  className="block w-full rounded-xl border border-zinc-800 bg-zinc-900/40 px-3 py-2.5 text-left text-sm text-zinc-300 transition hover:border-cyan-700/60 hover:text-zinc-100 disabled:opacity-50"
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`flex flex-col ${m.role === "user" ? "items-end" : "items-start"}`}>
            {m.role === "assistant" && <ResearchPills entries={m.researched ?? []} />}
            <div
              className={`max-w-[85%] rounded-2xl px-3.5 py-2 text-sm leading-relaxed ${
                m.role === "user"
                  ? "bg-cyan-500 text-zinc-950"
                  : "border border-zinc-800 bg-zinc-900/60 text-zinc-200"
              }`}
            >
              {m.text}
            </div>
          </div>
        ))}

        {busy && (
          <div className="flex flex-col items-start">
            <ResearchPills entries={progress?.researched ?? []} />
            <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 px-3.5 py-2 text-sm text-zinc-400">
              <span className="dot">●</span> {progress?.status ?? "thinking…"}
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      {started && (
        <div className="px-3 pb-1 pt-2">
          <div className="flex flex-wrap gap-1.5">
            {HINTS.map((h) => (
              <button
                key={h}
                onClick={() => setText(h.replace(/"/g, ""))}
                disabled={busy}
                className="rounded-full border border-zinc-800 px-2 py-0.5 text-[10px] text-zinc-500 transition hover:border-zinc-600 hover:text-zinc-300 disabled:opacity-50"
              >
                {h}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="border-t border-zinc-800 p-3">
        <div className="flex items-end gap-2 rounded-2xl border border-zinc-800 bg-zinc-900/40 p-2 transition focus-within:border-cyan-500/50">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            rows={1}
            placeholder={started ? "Refine it… (Enter to send)" : "Describe a theme…"}
            className="max-h-28 flex-1 resize-none bg-transparent px-1.5 py-1 text-sm text-zinc-100 outline-none placeholder:text-zinc-600"
          />
          <button
            onClick={submit}
            disabled={busy || !text.trim()}
            className="rounded-xl bg-cyan-500 px-3.5 py-2 text-sm font-semibold text-zinc-950 transition hover:bg-cyan-400 disabled:opacity-40"
          >
            ↑
          </button>
        </div>
      </div>
    </div>
  );
}
