"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const COLORS = ["#22d3ee", "#34d399", "#a78bfa", "#f59e0b", "#f472b6", "#60a5fa", "#fb7185", "#facc15"];

export function AllocationChart({ weights }: { weights: Record<string, number> }) {
  const data = Object.entries(weights)
    .filter(([, w]) => w > 0.001)
    .map(([symbol, w]) => ({ symbol, weight: +(w * 100).toFixed(1) }))
    .sort((a, b) => b.weight - a.weight);

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 10, right: 16, bottom: 8, left: 4 }}>
        <CartesianGrid stroke="#27272a" vertical={false} />
        <XAxis dataKey="symbol" stroke="#a1a1aa" tick={{ fontSize: 11 }} interval={0} angle={-30} textAnchor="end" height={50} />
        <YAxis stroke="#a1a1aa" tick={{ fontSize: 11 }} unit="%" />
        <Tooltip
          cursor={{ fill: "#27272a55" }}
          contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", borderRadius: 8, color: "#e4e4e7" }}
          formatter={(value) => `${value}%`}
        />
        <Bar dataKey="weight" radius={[4, 4, 0, 0]}>
          {data.map((_, i) => (
            <Cell key={i} fill={COLORS[i % COLORS.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
