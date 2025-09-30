"use client"

import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

const data = [
  { day: "Mon", verified: 420, fraud: 18 },
  { day: "Tue", verified: 512, fraud: 22 },
  { day: "Wed", verified: 480, fraud: 19 },
  { day: "Thu", verified: 530, fraud: 20 },
  { day: "Fri", verified: 600, fraud: 24 },
  { day: "Sat", verified: 300, fraud: 12 },
  { day: "Sun", verified: 250, fraud: 10 },
]

export function VerificationTrends() {
  return (
    <div className="rounded-lg border p-4 bg-card">
      <p className="font-medium">Weekly Verification Trends</p>
      <div className="h-56 mt-4">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="colorVerified" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="oklch(var(--color-primary))" stopOpacity={0.8} />
                <stop offset="95%" stopColor="oklch(var(--color-primary))" stopOpacity={0.1} />
              </linearGradient>
              <linearGradient id="colorFraud" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="oklch(var(--color-accent))" stopOpacity={0.8} />
                <stop offset="95%" stopColor="oklch(var(--color-accent))" stopOpacity={0.1} />
              </linearGradient>
            </defs>
            <XAxis dataKey="day" stroke="oklch(var(--color-muted-foreground))" />
            <YAxis stroke="oklch(var(--color-muted-foreground))" />
            <Tooltip
              contentStyle={{
                background: "oklch(var(--color-card))",
                border: "1px solid oklch(var(--color-border))",
                color: "oklch(var(--color-foreground))",
              }}
            />
            <Area type="monotone" dataKey="verified" stroke="oklch(var(--color-primary))" fill="url(#colorVerified)" />
            <Area type="monotone" dataKey="fraud" stroke="oklch(var(--color-accent))" fill="url(#colorFraud)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
