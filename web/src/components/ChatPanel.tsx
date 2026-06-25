"use client";

import { useEffect, useRef, useState } from "react";

export interface Msg {
  role: "user" | "assistant";
  text: string;
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
}: {
  messages: Msg[];
  onSend: (text: string) => void;
  busy: boolean;
  started: boolean;
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
          <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
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
          <div className="flex justify-start">
            <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 px-3.5 py-2 text-sm text-zinc-400">
              <span className="dot">●</span> thinking…
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
