"use client";

import {
  CartesianGrid,
  ReferenceDot,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { FrontierPoint } from "../lib/api";

export function FrontierChart({
  frontier,
  chosen,
}: {
  frontier: FrontierPoint[];
  chosen: { volatility: number; expected_return: number };
}) {
  const data = frontier.map((p) => ({
    x: +(p.volatility * 100).toFixed(2),
    y: +(p.expected_return * 100).toFixed(2),
  }));
  const cx = +(chosen.volatility * 100).toFixed(2);
  const cy = +(chosen.expected_return * 100).toFixed(2);

  return (
    <ResponsiveContainer width="100%" height={260}>
      <ScatterChart margin={{ top: 10, right: 24, bottom: 24, left: 4 }}>
        <CartesianGrid stroke="#27272a" />
        <XAxis
          type="number"
          dataKey="x"
          name="Risk"
          unit="%"
          stroke="#a1a1aa"
          tick={{ fontSize: 11 }}
          label={{ value: "Risk (annual volatility %)", position: "insideBottom", offset: -12, fill: "#71717a", fontSize: 11 }}
        />
        <YAxis
          type="number"
          dataKey="y"
          name="Return"
          unit="%"
          stroke="#a1a1aa"
          tick={{ fontSize: 11 }}
          label={{ value: "Expected return %", angle: -90, position: "insideLeft", fill: "#71717a", fontSize: 11 }}
        />
        <Tooltip
          contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", borderRadius: 8, color: "#e4e4e7" }}
          formatter={(value) => `${value}%`}
        />
        <Scatter data={data} line={{ stroke: "#22d3ee" }} fill="#22d3ee" />
        <ReferenceDot x={cx} y={cy} r={6} fill="#f59e0b" stroke="#fff" strokeWidth={1.5} ifOverflow="extendDomain" />
      </ScatterChart>
    </ResponsiveContainer>
  );
}
