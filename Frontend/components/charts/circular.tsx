"use client"

import { Pie, PieChart, Cell, ResponsiveContainer, Tooltip, Legend } from "recharts"

const data = [
  { name: "Verified", value: 940 },
  { name: "Potential Fraud", value: 60 },
]

const COLORS = ["oklch(var(--color-primary))", "oklch(var(--color-accent))"]

export function VerificationBreakdown() {
  return (
    <div className="rounded-lg border p-4 bg-card">
      <p className="font-medium">Verification Breakdown</p>
      <div className="h-56 mt-4">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Tooltip
              contentStyle={{
                background: "oklch(var(--color-card))",
                border: "1px solid oklch(var(--color-border))",
                color: "oklch(var(--color-foreground))",
              }}
            />
            <Legend />
            <Pie
              data={data}
              dataKey="value"
              nameKey="name"
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={85}
              stroke="oklch(var(--color-border))"
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${entry.name}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
