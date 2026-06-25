"use client";

import { useEffect, useRef, useState } from "react";
import { AllocationChart } from "../components/AllocationChart";
import { Findings } from "../components/Findings";
import { FrontierChart } from "../components/FrontierChart";
import { HoldingsTable } from "../components/HoldingsTable";
import {
  api,
  money,
  money2,
  pct,
  type ApproveResult,
  type ClarifyResponse,
  type Proposal,
} from "../lib/api";

const EXAMPLES = [
  "Nuclear energy and uranium mining, including the supply chain — lower risk, diversified",
  "Quantum computing and the companies that supply its cooling hardware",
  "Defense and aerospace primes, follow where the smart money is going",
  "The AI data-center buildout: chips, power, and cooling",
];

const BUILD_STEPS = [
  "Searching real SEC filings…",
  "Tracing supply-chain connections…",
  "Optimizing weights with real math…",
  "Running the safety circuit breaker…",
];

type Stage = "compose" | "clarify" | "result";

export default function Home() {
  const [stage, setStage] = useState<Stage>("compose");
  const [prompt, setPrompt] = useState(EXAMPLES[0]);
  const [cash, setCash] = useState(10000);

  const [clarify, setClarify] = useState<ClarifyResponse | null>(null);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [excluded, setExcluded] = useState<string[]>([]);

  const [proposal, setProposal] = useState<Proposal | null>(null);
  const [approved, setApproved] = useState<ApproveResult | null>(null);

  const [thinking, setThinking] = useState(false);
  const [building, setBuilding] = useState(false);
  const [pruning, setPruning] = useState<string | null>(null);
  const [approving, setApproving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onContinue() {
    setThinking(true);
    setError(null);
    try {
      const c = await api.clarify(prompt);
      setClarify(c);
      const init: Record<string, string> = {};
      for (const q of c.questions) init[q.id] = q.default ?? q.options[0].value;
      setAnswers(init);
      setExcluded([]);
      setStage("clarify");
    } catch (e) {
      setError(String(e));
    } finally {
      setThinking(false);
    }
  }

  async function build(opts: { answers?: Record<string, string>; excluded?: string[] } = {}) {
    const a = opts.answers ?? answers;
    const ex = opts.excluded ?? excluded;
    const res = await api.compile(prompt, {
      initial_cash: cash,
      lookback: a.lookback ?? "2y",
      answers: { ...a, exclude_symbols: ex },
    });
    setProposal(res);
    setApproved(null);
  }

  async function onBuild() {
    setBuilding(true);
    setError(null);
    setStage("result");
    try {
      await build();
    } catch (e) {
      setError(String(e));
    } finally {
      setBuilding(false);
    }
  }

  async function onPrune(symbol: string) {
    setPruning(symbol);
    setError(null);
    const next = [...excluded, symbol];
    setExcluded(next);
    try {
      await build({ excluded: next });
    } catch (e) {
      setError(String(e));
    } finally {
      setPruning(null);
    }
  }

  async function onChangeLookback(lb: string) {
    const next = { ...answers, lookback: lb };
    setAnswers(next);
    setBuilding(true);
    try {
      await build({ answers: next });
    } catch (e) {
      setError(String(e));
    } finally {
      setBuilding(false);
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

  function restart() {
    setStage("compose");
    setProposal(null);
    setApproved(null);
    setExcluded([]);
    setError(null);
  }

  return (
    <main className="mx-auto w-full max-w-5xl flex-1 px-5 py-8 sm:py-12">
      <Header stage={stage} onHome={restart} />

      {error && (
        <div className="mt-4 rounded-xl border border-red-900/60 bg-red-950/30 p-3 text-sm text-red-300">
          {error}
          <div className="mt-1 text-xs text-red-400/70">
            Backend trouble? Make sure <code>si serve</code> is running on :8000.
          </div>
        </div>
      )}

      {stage === "compose" && (
        <Compose
          prompt={prompt}
          setPrompt={setPrompt}
          onContinue={onContinue}
          thinking={thinking}
        />
      )}

      {stage === "clarify" && clarify && (
        <Clarify
          data={clarify}
          answers={answers}
          setAnswer={(id, v) => setAnswers((p) => ({ ...p, [id]: v }))}
          cash={cash}
          setCash={setCash}
          onBack={() => setStage("compose")}
          onBuild={onBuild}
        />
      )}

      {stage === "result" && (
        <Result
          building={building}
          proposal={proposal}
          approved={approved}
          approving={approving}
          pruning={pruning}
          cash={cash}
          lookback={answers.lookback ?? "2y"}
          focus={clarify?.focus}
          onApprove={onApprove}
          onPrune={onPrune}
          onChangeLookback={onChangeLookback}
          onRefine={() => setStage("clarify")}
        />
      )}

      <footer className="mt-14 text-center text-xs text-zinc-600">
        Paper trading · not investment advice · Robinhood Agentic MCP execution comes later
      </footer>
    </main>
  );
}

/* ------------------------------------------------------------------ Header */
function Header({ stage, onHome }: { stage: Stage; onHome: () => void }) {
  const steps: { id: Stage; label: string }[] = [
    { id: "compose", label: "Thesis" },
    { id: "clarify", label: "Refine" },
    { id: "result", label: "Portfolio" },
  ];
  const idx = steps.findIndex((s) => s.id === stage);
  return (
    <header className="flex items-center justify-between">
      <button onClick={onHome} className="text-left">
        <h1 className="text-xl font-semibold tracking-tight text-zinc-50">
          smart<span className="text-cyan-400">·</span>investing
        </h1>
      </button>
      <nav className="flex items-center gap-1.5 text-xs">
        {steps.map((s, i) => (
          <span key={s.id} className="flex items-center gap-1.5">
            <span
              className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 transition ${
                i === idx
                  ? "bg-cyan-500/15 text-cyan-300"
                  : i < idx
                    ? "text-zinc-400"
                    : "text-zinc-600"
              }`}
            >
              <span
                className={`grid h-4 w-4 place-items-center rounded-full text-[10px] ${
                  i <= idx ? "bg-cyan-500 text-zinc-950" : "bg-zinc-800 text-zinc-500"
                }`}
              >
                {i < idx ? "✓" : i + 1}
              </span>
              {s.label}
            </span>
            {i < steps.length - 1 && <span className="text-zinc-700">›</span>}
          </span>
        ))}
      </nav>
    </header>
  );
}

/* ----------------------------------------------------------------- Compose */
function Compose({
  prompt,
  setPrompt,
  onContinue,
  thinking,
}: {
  prompt: string;
  setPrompt: (s: string) => void;
  onContinue: () => void;
  thinking: boolean;
}) {
  return (
    <section className="rise mt-10">
      <h2 className="text-3xl font-semibold tracking-tight text-zinc-50 sm:text-4xl">
        What do you want to invest in?
      </h2>
      <p className="mt-3 max-w-xl text-[15px] leading-relaxed text-zinc-400">
        Describe a theme in plain English. I&rsquo;ll ask a couple of quick questions, then search real SEC
        filings, find the supply-chain names, and build an optimized portfolio you can actually understand.
      </p>

      <div className="mt-7 rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4 shadow-2xl shadow-black/30 transition focus-within:border-cyan-500/50">
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={3}
          autoFocus
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) onContinue();
          }}
          className="w-full resize-none bg-transparent text-[15px] leading-relaxed text-zinc-100 outline-none placeholder:text-zinc-600"
          placeholder="e.g. Invest in nuclear power and the uranium supply chain, keep it lower-risk…"
        />
        <div className="mt-3 flex items-center justify-between gap-3">
          <span className="text-xs text-zinc-600">⌘↵ to continue</span>
          <button
            onClick={onContinue}
            disabled={thinking || !prompt.trim()}
            className="rounded-xl bg-cyan-500 px-5 py-2.5 text-sm font-semibold text-zinc-950 transition hover:bg-cyan-400 disabled:opacity-50"
          >
            {thinking ? "Reading your thesis…" : "Continue →"}
          </button>
        </div>
      </div>

      <div className="mt-6">
        <div className="text-xs font-medium uppercase tracking-wide text-zinc-600">Try one</div>
        <div className="mt-2 flex flex-col gap-2">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              onClick={() => setPrompt(ex)}
              className="rounded-xl border border-zinc-800 bg-zinc-900/30 px-4 py-2.5 text-left text-sm text-zinc-300 transition hover:border-cyan-700/60 hover:text-zinc-100"
            >
              {ex}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ----------------------------------------------------------------- Clarify */
function Clarify({
  data,
  answers,
  setAnswer,
  cash,
  setCash,
  onBack,
  onBuild,
}: {
  data: ClarifyResponse;
  answers: Record<string, string>;
  setAnswer: (id: string, v: string) => void;
  cash: number;
  setCash: (n: number) => void;
  onBack: () => void;
  onBuild: () => void;
}) {
  return (
    <section className="rise mt-8 space-y-5">
      <div className="flex items-start gap-3 rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4">
        <span className="mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-full bg-cyan-500/15 text-cyan-300">
          ✦
        </span>
        <p className="text-[15px] leading-relaxed text-zinc-100">{data.interpretation}</p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        {data.questions.map((q) => (
          <div key={q.id} className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4">
            <div className="text-sm font-medium text-zinc-100">{q.question}</div>
            <div className="mt-1 text-xs leading-relaxed text-zinc-500">{q.help}</div>
            <div className="mt-3 flex flex-wrap gap-2">
              {q.options.map((o) => {
                const active = answers[q.id] === o.value;
                return (
                  <button
                    key={o.value}
                    onClick={() => setAnswer(q.id, o.value)}
                    title={o.hint}
                    className={`rounded-xl border px-3 py-1.5 text-xs transition ${
                      active
                        ? "border-cyan-400 bg-cyan-500/15 text-cyan-200"
                        : "border-zinc-700 bg-zinc-900/40 text-zinc-300 hover:border-zinc-500"
                    }`}
                  >
                    {o.label}
                  </button>
                );
              })}
            </div>
            {answers[q.id] && (
              <div className="mt-2 text-[11px] text-zinc-500">
                {q.options.find((o) => o.value === answers[q.id])?.hint}
              </div>
            )}
          </div>
        ))}

        <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4">
          <div className="text-sm font-medium text-zinc-100">Starting cash</div>
          <div className="mt-1 text-xs leading-relaxed text-zinc-500">
            Paper money to allocate across this portfolio.
          </div>
          <div className="mt-3 flex items-center gap-2">
            <span className="text-zinc-400">$</span>
            <input
              type="number"
              value={cash}
              min={500}
              step={500}
              onChange={(e) => setCash(+e.target.value)}
              className="w-32 rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-sm text-zinc-100 outline-none focus:border-cyan-500"
            />
            <div className="ml-1 flex gap-1.5">
              {[5000, 10000, 25000].map((v) => (
                <button
                  key={v}
                  onClick={() => setCash(v)}
                  className="rounded-lg border border-zinc-700 px-2 py-1 text-[11px] text-zinc-400 hover:border-zinc-500"
                >
                  {money(v)}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="flex items-center justify-between">
        <button onClick={onBack} className="text-sm text-zinc-400 hover:text-zinc-200">
          ← Edit thesis
        </button>
        <button
          onClick={onBuild}
          className="rounded-xl bg-cyan-500 px-6 py-2.5 text-sm font-semibold text-zinc-950 transition hover:bg-cyan-400"
        >
          Build my portfolio →
        </button>
      </div>
      <p className="text-center text-xs text-zinc-600">
        We&rsquo;ll decide the right number of holdings for you.
      </p>
    </section>
  );
}

/* ------------------------------------------------------------------ Result */
const LOOKBACKS = [
  { value: "1y", label: "1Y" },
  { value: "2y", label: "2Y" },
  { value: "3y", label: "3Y" },
  { value: "5y", label: "5Y" },
];

function Result({
  building,
  proposal,
  approved,
  approving,
  pruning,
  cash,
  lookback,
  focus,
  onApprove,
  onPrune,
  onChangeLookback,
  onRefine,
}: {
  building: boolean;
  proposal: Proposal | null;
  approved: ApproveResult | null;
  approving: boolean;
  pruning: string | null;
  cash: number;
  lookback: string;
  focus?: string;
  onApprove: () => void;
  onPrune: (s: string) => void;
  onChangeLookback: (lb: string) => void;
  onRefine: () => void;
}) {
  if (building && !proposal) return <BuildingState />;
  if (!proposal) return null;
  const o = proposal.optimization;

  return (
    <section className="rise mt-8 space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <div className="text-xs uppercase tracking-wide text-cyan-400/70">Your portfolio</div>
          <h2 className="mt-0.5 text-2xl font-semibold tracking-tight text-zinc-50">
            {focus || proposal.spec.themes[0] || "Custom thesis"}
          </h2>
        </div>
        <button onClick={onRefine} className="text-sm text-zinc-400 hover:text-zinc-200">
          ← Adjust answers
        </button>
      </div>

      {proposal.explanation && <Findings explanation={proposal.explanation} />}

      {proposal.blocked && (
        <div className="rounded-xl border border-amber-800/60 bg-amber-950/20 p-4 text-sm text-amber-200">
          <div className="font-medium">Circuit breaker blocked execution</div>
          <ul className="mt-1.5 list-disc space-y-0.5 pl-5 text-xs text-amber-300/80">
            {proposal.violations.filter((v) => v.severity === "fatal").map((v) => (
              <li key={v.code}>{v.message}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Metric label="Expected return" value={pct(o.expected_return)} />
        <Metric label="Risk (volatility)" value={pct(o.volatility)} />
        <Metric label="Sharpe ratio" value={o.sharpe.toFixed(2)} highlight />
        {proposal.backtest ? (
          <Metric label={`Backtest CAGR · ${lookback}`} value={pct(proposal.backtest.cagr)} />
        ) : (
          <Metric label="Holdings" value={String(Object.values(proposal.target_weights).filter((w) => w > 0.005).length)} />
        )}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Efficient frontier" subtitle="amber dot = your optimized portfolio">
          <FrontierChart frontier={o.frontier} chosen={{ volatility: o.volatility, expected_return: o.expected_return }} />
        </Card>
        <Card title="Allocation">
          <AllocationChart weights={proposal.target_weights} />
        </Card>
      </div>

      <Card
        title={`Holdings — ${Object.values(proposal.target_weights).filter((w) => w > 0.005).length} names`}
        subtitle="hover a row to remove a name and rebuild"
        right={
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-500">Analyze</span>
            <Segmented value={lookback} options={LOOKBACKS} onChange={onChangeLookback} disabled={building} />
          </div>
        }
      >
        {building ? (
          <div className="py-6 text-center text-sm text-zinc-500">
            <span className="dot">●</span> Rebuilding…
          </div>
        ) : (
          <HoldingsTable proposal={proposal} onPrune={onPrune} pruning={pruning} />
        )}
      </Card>

      <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-5">
        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={onApprove}
            disabled={approving || proposal.trades.length === 0}
            className="rounded-xl bg-emerald-500 px-5 py-2.5 text-sm font-semibold text-zinc-950 transition hover:bg-emerald-400 disabled:opacity-50"
          >
            {approving ? "Executing…" : `Approve & execute ${proposal.trades.length} orders (paper)`}
          </button>
          <span className="text-xs text-zinc-500">
            {money2(cash)} starting · {proposal.price_source}
          </span>
        </div>
        {approved && (
          <div className="rise mt-4 rounded-xl border border-emerald-900/60 bg-emerald-950/20 p-4 text-sm text-emerald-200">
            ✓ Executed {approved.filled} orders ({approved.rejected} rejected). Cash left {money2(approved.account.cash)} ·
            realized P&amp;L {money2(approved.realized_pnl)}.
            <div className="mt-1.5 text-xs text-emerald-300/70">
              Holdings: {Object.values(approved.account.positions).map((p) => `${p.symbol} ${p.quantity.toFixed(2)}`).join(" · ")}
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

/* ------------------------------------------------------------- Build state */
function BuildingState() {
  const [i, setI] = useState(0);
  const ref = useRef(0);
  useEffect(() => {
    const t = setInterval(() => {
      ref.current = Math.min(ref.current + 1, BUILD_STEPS.length - 1);
      setI(ref.current);
    }, 1100);
    return () => clearInterval(t);
  }, []);
  return (
    <div className="rise mt-10 rounded-2xl border border-zinc-800 bg-zinc-900/40 p-8">
      <div className="flex items-center gap-3">
        <span className="grid h-9 w-9 place-items-center rounded-full bg-cyan-500/15 text-cyan-300">
          <span className="dot">●</span>
        </span>
        <div className="text-lg font-medium text-zinc-100">Building your portfolio</div>
      </div>
      <ul className="mt-5 space-y-2.5">
        {BUILD_STEPS.map((s, idx) => (
          <li
            key={s}
            className={`flex items-center gap-3 text-sm transition ${
              idx < i ? "text-zinc-500" : idx === i ? "text-zinc-100" : "text-zinc-700"
            }`}
          >
            <span
              className={`grid h-5 w-5 place-items-center rounded-full text-[10px] ${
                idx < i ? "bg-emerald-500/20 text-emerald-400" : idx === i ? "bg-cyan-500/20 text-cyan-300" : "bg-zinc-800 text-zinc-600"
              }`}
            >
              {idx < i ? "✓" : idx === i ? <span className="dot">●</span> : idx + 1}
            </span>
            {s}
          </li>
        ))}
      </ul>
    </div>
  );
}

/* ----------------------------------------------------------------- atoms */
function Metric({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
      <div className="text-xs text-zinc-500">{label}</div>
      <div className={`mt-1 text-xl font-semibold tabular-nums ${highlight ? "text-cyan-400" : "text-zinc-100"}`}>
        {value}
      </div>
    </div>
  );
}

function Card({
  title,
  subtitle,
  right,
  children,
}: {
  title: string;
  subtitle?: string;
  right?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-5">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-medium text-zinc-200">{title}</h2>
          {subtitle && <p className="mt-0.5 text-xs text-zinc-500">{subtitle}</p>}
        </div>
        {right}
      </div>
      {children}
    </div>
  );
}

function Segmented({
  value,
  options,
  onChange,
  disabled,
}: {
  value: string;
  options: { value: string; label: string }[];
  onChange: (v: string) => void;
  disabled?: boolean;
}) {
  return (
    <div className="inline-flex rounded-lg border border-zinc-700 bg-zinc-950 p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          disabled={disabled}
          onClick={() => onChange(o.value)}
          className={`rounded-md px-2.5 py-1 text-xs transition disabled:opacity-50 ${
            value === o.value ? "bg-cyan-500 text-zinc-950" : "text-zinc-400 hover:text-zinc-200"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
