"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type GraphNeighbor } from "../lib/api";

/* A node placed on the canvas. The theme sits at the origin; everything else is
   laid out by a small force relaxation so expansions stay legible. */
interface Node {
  symbol: string;
  name: string;
  x: number;
  y: number;
  relevance: number;
  depth: number; // hops from the theme
  fixed?: boolean; // the theme anchor
}
interface Edge {
  a: string;
  b: string;
  rel: string; // supplies | competes | co_mention
  dir: GraphNeighbor["direction"];
}

const THEME_ID = "__theme__";

const REL_STYLE: Record<string, { stroke: string; dash?: string }> = {
  supplies: { stroke: "#22d3ee" },
  competes: { stroke: "#f59e0b", dash: "5 4" },
  co_mention: { stroke: "#52525b", dash: "2 4" },
};

function relax(nodes: Node[], edges: Edge[]): Node[] {
  const pos = new Map(nodes.map((n) => [n.symbol, { x: n.x, y: n.y }]));
  const REP = 14000;
  const L = 110;
  const K = 0.03;
  const G = 0.012;
  for (let iter = 0; iter < 220; iter++) {
    const disp = new Map(nodes.map((n) => [n.symbol, { x: 0, y: 0 }]));
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const pa = pos.get(nodes[i].symbol)!;
        const pb = pos.get(nodes[j].symbol)!;
        let dx = pa.x - pb.x;
        let dy = pa.y - pb.y;
        const d2 = dx * dx + dy * dy || 0.01;
        const d = Math.sqrt(d2);
        const f = REP / d2;
        dx = (dx / d) * f;
        dy = (dy / d) * f;
        disp.get(nodes[i].symbol)!.x += dx;
        disp.get(nodes[i].symbol)!.y += dy;
        disp.get(nodes[j].symbol)!.x -= dx;
        disp.get(nodes[j].symbol)!.y -= dy;
      }
    }
    for (const e of edges) {
      const pa = pos.get(e.a);
      const pb = pos.get(e.b);
      if (!pa || !pb) continue;
      const dx = pa.x - pb.x;
      const dy = pa.y - pb.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 0.01;
      const f = (d - L) * K;
      const fx = (dx / d) * f;
      const fy = (dy / d) * f;
      disp.get(e.a)!.x -= fx;
      disp.get(e.a)!.y -= fy;
      disp.get(e.b)!.x += fx;
      disp.get(e.b)!.y += fy;
    }
    for (const n of nodes) {
      const p = pos.get(n.symbol)!;
      const dd = disp.get(n.symbol)!;
      dd.x -= p.x * G;
      dd.y -= p.y * G;
      if (n.fixed) continue;
      const step = 6;
      p.x += Math.max(-step, Math.min(step, dd.x));
      p.y += Math.max(-step, Math.min(step, dd.y));
    }
  }
  return nodes.map((n) => ({ ...n, ...pos.get(n.symbol)! }));
}

export function GraphCanvas({
  theme,
  pinned,
  onAdd,
  onRemove,
  busy,
}: {
  theme: string;
  pinned: string[];
  onAdd: (symbol: string) => void;
  onRemove: (symbol: string) => void;
  busy: boolean;
}) {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [selected, setSelected] = useState<string | null>(null);
  const [loading, setLoading] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);
  const seededFor = useRef<string>("");
  const pinnedSet = useMemo(() => new Set(pinned.map((s) => s.toUpperCase())), [pinned]);

  // (re)seed when the theme changes
  useEffect(() => {
    if (!theme || theme === seededFor.current) return;
    seededFor.current = theme;
    let cancelled = false;
    (async () => {
      setError(null);
      try {
        const { nodes: seeds } = await api.graphSearch(theme, 8);
        if (cancelled) return;
        const center: Node = { symbol: THEME_ID, name: theme, x: 0, y: 0, relevance: 1, depth: 0, fixed: true };
        const ring = seeds.map((s, i) => {
          const ang = (i / Math.max(seeds.length, 1)) * Math.PI * 2;
          return {
            symbol: s.symbol,
            name: s.name,
            x: Math.cos(ang) * 170,
            y: Math.sin(ang) * 170,
            relevance: s.relevance,
            depth: 1,
          } as Node;
        });
        const es: Edge[] = seeds.map((s) => ({ a: THEME_ID, b: s.symbol, rel: "supplies", dir: "downstream" }));
        setEdges(es);
        setNodes(relax([center, ...ring], es));
        setExpanded(new Set());
        setSelected(null);
      } catch (e) {
        if (!cancelled) setError(String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [theme]);

  const expand = useCallback(
    async (symbol: string) => {
      if (symbol === THEME_ID || expanded.has(symbol)) {
        setSelected(symbol);
        return;
      }
      setSelected(symbol);
      setLoading((s) => new Set(s).add(symbol));
      try {
        const { neighbors } = await api.graphNeighbors(symbol, theme, 10);
        setNodes((prev) => {
          const byId = new Map(prev.map((n) => [n.symbol, n]));
          const parent = byId.get(symbol)!;
          const fresh = neighbors.filter((nb) => !byId.has(nb.symbol));
          const added: Node[] = fresh.map((nb, i) => {
            const ang = (i / Math.max(fresh.length, 1)) * Math.PI * 2;
            return {
              symbol: nb.symbol,
              name: nb.name,
              x: parent.x + Math.cos(ang) * 90 + (parent.x === 0 ? 0 : parent.x * 0.15),
              y: parent.y + Math.sin(ang) * 90,
              relevance: nb.relevance,
              depth: parent.depth + 1,
            };
          });
          const newEdges: Edge[] = neighbors.map((nb) => ({
            a: nb.direction === "upstream" ? nb.symbol : symbol,
            b: nb.direction === "upstream" ? symbol : nb.symbol,
            rel: nb.rel,
            dir: nb.direction,
          }));
          setEdges((pe) => {
            const have = new Set(pe.map((e) => `${e.a}>${e.b}`));
            const merged = [...pe];
            for (const e of newEdges) if (!have.has(`${e.a}>${e.b}`) && e.a !== e.b) merged.push(e);
            const allNodes = [...prev, ...added];
            queueMicrotask(() => setNodes(relax(allNodes, merged)));
            return merged;
          });
          return [...prev, ...added];
        });
        setExpanded((s) => new Set(s).add(symbol));
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading((s) => {
          const n = new Set(s);
          n.delete(symbol);
          return n;
        });
      }
    },
    [expanded, theme],
  );

  const view = useMemo(() => {
    if (nodes.length === 0) return "-260 -200 520 400";
    const xs = nodes.map((n) => n.x);
    const ys = nodes.map((n) => n.y);
    const pad = 90;
    const minX = Math.min(...xs) - pad;
    const minY = Math.min(...ys) - pad;
    const w = Math.max(...xs) - minX + pad;
    const h = Math.max(...ys) - minY + pad;
    return `${minX} ${minY} ${w} ${h}`;
  }, [nodes]);

  const sel = nodes.find((n) => n.symbol === selected) || null;

  if (!theme) {
    return (
      <div className="grid h-full place-items-center text-center text-sm text-zinc-600">
        <div>
          <div className="text-3xl">◌</div>
          <p className="mt-2 max-w-xs">Describe a theme in the chat — the supply-chain map appears here.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative h-full w-full overflow-hidden">
      {error && (
        <div className="absolute left-3 top-3 z-10 rounded-lg border border-red-900/60 bg-red-950/40 px-2 py-1 text-xs text-red-300">
          {error}
        </div>
      )}
      <div className="absolute right-3 top-3 z-10 flex flex-col gap-1 rounded-lg border border-zinc-800 bg-zinc-950/70 px-2.5 py-2 text-[10px] text-zinc-400 backdrop-blur">
        <Legend color="#22d3ee" label="supplies →" />
        <Legend color="#f59e0b" label="competes" dash />
        <Legend color="#52525b" label="co-mentioned" dash />
        <div className="mt-1 flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-emerald-400" /> in portfolio
        </div>
        <div className="text-zinc-600">click a node to walk deeper</div>
      </div>

      <svg viewBox={view} className="h-full w-full" preserveAspectRatio="xMidYMid meet">
        <defs>
          <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto" markerUnits="userSpaceOnUse">
            <path d="M0,0 L7,3 L0,6 Z" fill="#22d3ee" opacity="0.7" />
          </marker>
        </defs>
        {edges.map((e, i) => {
          const a = nodes.find((n) => n.symbol === e.a);
          const b = nodes.find((n) => n.symbol === e.b);
          if (!a || !b) return null;
          const st = REL_STYLE[e.rel] || REL_STYLE.co_mention;
          return (
            <line
              key={i}
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              stroke={st.stroke}
              strokeWidth={e.rel === "supplies" ? 1.4 : 1}
              strokeDasharray={st.dash}
              opacity={0.45}
              markerEnd={e.rel === "supplies" ? "url(#arrow)" : undefined}
            />
          );
        })}
        {nodes.map((n) => (
          <NodeGlyph
            key={n.symbol}
            n={n}
            pinned={pinnedSet.has(n.symbol)}
            selected={n.symbol === selected}
            expanded={expanded.has(n.symbol)}
            loading={loading.has(n.symbol)}
            busy={busy}
            onClick={() => expand(n.symbol)}
            onAdd={() => onAdd(n.symbol)}
            onRemove={() => onRemove(n.symbol)}
          />
        ))}
      </svg>

      {sel && sel.symbol !== THEME_ID && (
        <div className="absolute bottom-3 left-3 z-10 max-w-[260px] rounded-xl border border-zinc-800 bg-zinc-950/85 p-3 text-xs backdrop-blur">
          <div className="flex items-center gap-2">
            <span className="rounded bg-zinc-800 px-1.5 py-0.5 font-semibold text-zinc-100">{sel.symbol}</span>
            <span className="truncate text-zinc-400">{sel.name}</span>
          </div>
          <div className="mt-1 text-zinc-500">relevance {(sel.relevance * 100).toFixed(0)}% · {sel.depth} hop{sel.depth === 1 ? "" : "s"} from theme</div>
          <div className="mt-2 flex gap-2">
            {pinnedSet.has(sel.symbol) ? (
              <button onClick={() => onRemove(sel.symbol)} disabled={busy} className="rounded-lg border border-zinc-700 px-2 py-1 text-zinc-300 hover:border-red-700 hover:text-red-300 disabled:opacity-50">
                Remove from portfolio
              </button>
            ) : (
              <button onClick={() => onAdd(sel.symbol)} disabled={busy} className="rounded-lg bg-cyan-500 px-2 py-1 font-medium text-zinc-950 hover:bg-cyan-400 disabled:opacity-50">
                + Add to portfolio
              </button>
            )}
            {!expanded.has(sel.symbol) && (
              <button onClick={() => expand(sel.symbol)} className="rounded-lg border border-zinc-700 px-2 py-1 text-zinc-300 hover:border-zinc-500">
                Expand
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function Legend({ color, label, dash }: { color: string; label: string; dash?: boolean }) {
  return (
    <div className="flex items-center gap-1.5">
      <svg width="18" height="6">
        <line x1="0" y1="3" x2="18" y2="3" stroke={color} strokeWidth="1.6" strokeDasharray={dash ? "3 3" : undefined} />
      </svg>
      {label}
    </div>
  );
}

function NodeGlyph({
  n,
  pinned,
  selected,
  expanded,
  loading,
  busy,
  onClick,
  onAdd,
  onRemove,
}: {
  n: Node;
  pinned: boolean;
  selected: boolean;
  expanded: boolean;
  loading: boolean;
  busy: boolean;
  onClick: () => void;
  onAdd: () => void;
  onRemove: () => void;
}) {
  const isTheme = n.symbol === THEME_ID;
  if (isTheme) {
    const label = n.name.length > 22 ? n.name.slice(0, 21) + "…" : n.name;
    const w = Math.max(80, label.length * 7.2 + 24);
    return (
      <g transform={`translate(${n.x},${n.y})`}>
        <rect x={-w / 2} y={-16} width={w} height={32} rx={16} fill="#0891b2" opacity={0.25} stroke="#22d3ee" />
        <text textAnchor="middle" dy="4" fontSize="12" fill="#a5f3fc" fontWeight={600}>
          {label}
        </text>
      </g>
    );
  }
  const w = Math.max(46, n.symbol.length * 9 + 22);
  const fill = pinned ? "#064e3b" : selected ? "#164e63" : "#18181b";
  const stroke = pinned ? "#34d399" : selected ? "#22d3ee" : "#3f3f46";
  return (
    <g transform={`translate(${n.x},${n.y})`} className="cursor-pointer" style={{ transition: "transform 0.4s ease" }}>
      <rect
        x={-w / 2}
        y={-15}
        width={w}
        height={30}
        rx={9}
        fill={fill}
        stroke={stroke}
        strokeWidth={selected || pinned ? 2 : 1.2}
        onClick={onClick}
      />
      <text textAnchor="middle" dy="4" fontSize="12" fill={pinned ? "#a7f3d0" : "#e4e4e7"} fontWeight={600} onClick={onClick} style={{ pointerEvents: "none" }}>
        {n.symbol}
      </text>
      {loading && <circle cx={w / 2} cy={-15} r={3} fill="#22d3ee" className="animate-pulse" />}
      {!expanded && !loading && (
        <text x={w / 2 - 2} y={-9} fontSize="9" fill="#71717a" style={{ pointerEvents: "none" }}>
          ⊕
        </text>
      )}
      {/* add / remove toggle */}
      <g
        transform={`translate(${w / 2 - 2},12)`}
        onClick={(ev) => {
          ev.stopPropagation();
          if (!busy) (pinned ? onRemove : onAdd)();
        }}
        className={busy ? "" : "cursor-pointer"}
      >
        <circle r={8} fill={pinned ? "#065f46" : "#0e7490"} stroke={pinned ? "#34d399" : "#22d3ee"} strokeWidth={1} />
        <text textAnchor="middle" dy="3.5" fontSize="11" fill="#ecfeff" fontWeight={700} style={{ pointerEvents: "none" }}>
          {pinned ? "−" : "+"}
        </text>
      </g>
    </g>
  );
}
