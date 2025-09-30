type StatCardProps = {
  title: string
  value: string
  caption?: string
}
function StatCard({ title, value, caption }: StatCardProps) {
  return (
    <div className="rounded-lg border p-4 bg-card">
      <p className="text-sm text-muted-foreground">{title}</p>
      <p className="text-2xl font-semibold mt-2">{value}</p>
      {caption && <p className="text-xs text-muted-foreground mt-1">{caption}</p>}
    </div>
  )
}

export function SummaryCards() {
  return (
    <div className="grid gap-4 md:grid-cols-4">
      <StatCard title="Total Verifications" value="12,487" caption="All time" />
      <StatCard title="Verified" value="11,932" caption="Past 30 days: 96%" />
      <StatCard title="Potential Fraud" value="555" caption="Past 30 days: 4%" />
      <StatCard title="Avg. Response" value="480ms" caption="P95: 1.2s" />
    </div>
  )
}
