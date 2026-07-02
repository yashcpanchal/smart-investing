"use client";

import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";

const COLORS = [
  "#22d3ee", "#34d399", "#a78bfa", "#f59e0b", "#f472b6",
  "#60a5fa", "#fb7185", "#facc15", "#2dd4bf", "#c084fc",
];

export function AllocationChart({ weights }: { weights: Record<string, number> }) {
  const data = Object.entries(weights)
    .filter(([, w]) => w > 0.001)
    .map(([symbol, w]) => ({ symbol, weight: +(w * 100).toFixed(1) }))
    .sort((a, b) => b.weight - a.weight);

  return (
    <div className="flex flex-col items-center gap-4 sm:flex-row">
      <ResponsiveContainer width="100%" height={220} className="!w-full sm:!w-1/2">
        <PieChart>
          <Pie
            data={data}
            dataKey="weight"
            nameKey="symbol"
            innerRadius={52}
            outerRadius={88}
            paddingAngle={2}
            stroke="#0c0d12"
            strokeWidth={2}
          >
            {data.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{ background: "#15161c", border: "1px solid #2a2c33", borderRadius: 10, color: "#e4e4e7" }}
            formatter={(value, name) => [`${value}%`, name as string]}
          />
        </PieChart>
      </ResponsiveContainer>
      <ul className="grid w-full grid-cols-2 gap-x-4 gap-y-1.5 text-xs sm:w-1/2 sm:grid-cols-1">
        {data.map((d, i) => (
          <li key={d.symbol} className="flex items-center gap-2">
            <span className="h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: COLORS[i % COLORS.length] }} />
            <span className="font-medium text-zinc-200">{d.symbol}</span>
            <span className="ml-auto tabular-nums text-zinc-400">{d.weight}%</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
