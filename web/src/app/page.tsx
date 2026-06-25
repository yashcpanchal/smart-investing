"use client";

import { useState } from "react";
import { ChatPanel, type Msg } from "../components/ChatPanel";
import { GraphCanvas } from "../components/GraphCanvas";
import { IndustryPanel } from "../components/IndustryPanel";
import { PortfolioPanel } from "../components/PortfolioPanel";
import { StocksPanel } from "../components/StocksPanel";
import { api, type ChatState, type Proposal } from "../lib/api";

type View = "map" | "portfolio" | "stocks" | "industry";
type Tab = "chat" | View;

const VIEWS: { id: View; label: string }[] = [
  { id: "map", label: "Supply chain" },
  { id: "portfolio", label: "Portfolio" },
  { id: "stocks", label: "Stocks" },
  { id: "industry", label: "Industry" },
];

export default function Home() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [proposal, setProposal] = useState<Proposal | null>(null);
  const [chatState, setChatState] = useState<ChatState | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("chat");
  const [visited, setVisited] = useState<Set<View>>(new Set(["map", "portfolio"]));

  async function send(text: string) {
    setMessages((m) => [...m, { role: "user", text }]);
    setBusy(true);
    setError(null);
    try {
      const r = await api.chat(text, sessionId);
      setSessionId(r.session_id);
      setChatState(r.state);
      if (r.proposal) {
        setProposal(r.proposal);
        if (tab === "chat") goto("map");
      }
      setMessages((m) => [...m, { role: "assistant", text: r.reply }]);
    } catch (e) {
      setError(String(e));
      setMessages((m) => [...m, { role: "assistant", text: "Something went wrong reaching the engine." }]);
    } finally {
      setBusy(false);
    }
  }

  function goto(v: View) {
    setTab(v);
    setVisited((s) => (s.has(v) ? s : new Set(s).add(v)));
  }

  const addToPortfolio = (s: string) => send(`Add ${s} to the portfolio`);
  const removeFromPortfolio = (s: string) => send(`Drop ${s} from the portfolio`);

  function reset() {
    setSessionId(null);
    setMessages([]);
    setProposal(null);
    setChatState(null);
    setError(null);
    setTab("chat");
    setVisited(new Set(["map", "portfolio"]));
  }

  const started = messages.length > 0;
  const theme = chatState?.search_theme || chatState?.theme || "";
  const pinned = chatState?.pinned ?? [];
  const holdings = proposal?.explanation?.holdings ?? [];
  const view: View = tab === "chat" ? "map" : tab; // desktop workspace always shows a view

  return (
    <div className="flex h-screen flex-col bg-zinc-950">
      <header className="flex shrink-0 items-center justify-between border-b border-zinc-800 px-5 py-2.5">
        <button onClick={reset} className="text-left">
          <h1 className="text-base font-semibold tracking-tight text-zinc-50">
            smart<span className="text-cyan-400">·</span>investing
          </h1>
        </button>
        <div className="flex items-center gap-3">
          {chatState && (
            <div className="hidden items-center gap-2 text-[11px] text-zinc-500 sm:flex">
              <Pill>{chatState.risk} risk</Pill>
              <Pill>{chatState.breadth}</Pill>
              <Pill>{chatState.lookback}</Pill>
              <Pill>{pinned.length} pinned</Pill>
            </div>
          )}
          <button onClick={reset} className="rounded-lg border border-zinc-800 px-2.5 py-1 text-xs text-zinc-400 hover:border-zinc-600 hover:text-zinc-200">
            New
          </button>
        </div>
      </header>

      {error && (
        <div className="shrink-0 border-b border-red-900/50 bg-red-950/30 px-5 py-1.5 text-xs text-red-300">
          {error} — is the backend running on :8000?
        </div>
      )}

      {/* mobile tab switcher (incl. chat) */}
      <div className="flex shrink-0 overflow-x-auto border-b border-zinc-800 lg:hidden">
        {(["chat", ...VIEWS.map((v) => v.id)] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => (t === "chat" ? setTab("chat") : goto(t))}
            className={`shrink-0 px-3.5 py-2 text-xs font-medium capitalize transition ${
              tab === t ? "border-b-2 border-cyan-400 text-cyan-300" : "text-zinc-500"
            }`}
          >
            {t === "chat" ? "Chat" : VIEWS.find((v) => v.id === t)?.label}
          </button>
        ))}
      </div>

      <main className="grid min-h-0 flex-1 lg:grid-cols-[360px_1fr]">
        {/* chat — persistent on desktop */}
        <section className={`${tab === "chat" ? "flex" : "hidden"} min-h-0 flex-col border-zinc-800 lg:flex lg:border-r`}>
          <ChatPanel messages={messages} onSend={send} busy={busy} started={started} />
        </section>

        {/* workspace */}
        <section className={`${tab === "chat" ? "hidden" : "flex"} min-h-0 flex-col lg:flex`}>
          {/* desktop view tabs */}
          <div className="hidden shrink-0 gap-1 border-b border-zinc-800 px-3 py-2 lg:flex">
            {VIEWS.map((v) => (
              <button
                key={v.id}
                onClick={() => goto(v.id)}
                className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                  view === v.id ? "bg-cyan-500/15 text-cyan-300" : "text-zinc-400 hover:text-zinc-200"
                }`}
              >
                {v.label}
              </button>
            ))}
          </div>

          <div className="relative min-h-0 flex-1">
            <Pane show={view === "map"}>
              <GraphCanvas theme={theme} pinned={pinned} onAdd={addToPortfolio} onRemove={removeFromPortfolio} busy={busy} />
            </Pane>
            <Pane show={view === "portfolio"}>
              <PortfolioPanel proposal={proposal} busy={busy} onRemove={removeFromPortfolio} />
            </Pane>
            {visited.has("stocks") && (
              <Pane show={view === "stocks"}>
                <StocksPanel holdings={holdings} theme={theme} />
              </Pane>
            )}
            {visited.has("industry") && (
              <Pane show={view === "industry"}>
                <IndustryPanel theme={theme} />
              </Pane>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

function Pane({ show, children }: { show: boolean; children: React.ReactNode }) {
  return <div className={`absolute inset-0 ${show ? "block" : "hidden"}`}>{children}</div>;
}

function Pill({ children }: { children: React.ReactNode }) {
  return <span className="rounded-full border border-zinc-800 bg-zinc-900/60 px-2 py-0.5">{children}</span>;
}
