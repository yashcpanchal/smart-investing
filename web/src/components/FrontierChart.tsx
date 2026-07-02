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
    <ResponsiveContainer width="100%" height={240}>
      <ScatterChart margin={{ top: 12, right: 20, bottom: 24, left: 0 }}>
        <CartesianGrid stroke="#1c1e25" />
        <XAxis
          type="number"
          dataKey="x"
          name="Risk"
          unit="%"
          stroke="#52555f"
          tick={{ fontSize: 11, fill: "#8a8d97" }}
          label={{ value: "Risk (annual volatility %)", position: "insideBottom", offset: -12, fill: "#6b6e78", fontSize: 11 }}
        />
        <YAxis
          type="number"
          dataKey="y"
          name="Return"
          unit="%"
          stroke="#52555f"
          tick={{ fontSize: 11, fill: "#8a8d97" }}
          label={{ value: "Expected return %", angle: -90, position: "insideLeft", fill: "#6b6e78", fontSize: 11 }}
        />
        <Tooltip
          contentStyle={{ background: "#15161c", border: "1px solid #2a2c33", borderRadius: 10, color: "#e4e4e7" }}
          formatter={(value) => `${value}%`}
        />
        <Scatter data={data} line={{ stroke: "#22d3ee", strokeWidth: 2 }} fill="#22d3ee" fillOpacity={0.5} />
        <ReferenceDot x={cx} y={cy} r={7} fill="#f59e0b" stroke="#fff" strokeWidth={2} ifOverflow="extendDomain" />
      </ScatterChart>
    </ResponsiveContainer>
  );
}
