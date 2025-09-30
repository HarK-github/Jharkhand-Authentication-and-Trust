import { SiteHeader } from "@/components/site-header"
import { SiteFooter } from "@/components/site-footer"
import { SummaryCards } from "@/components/admin/summary-cards"
import { VerificationTrends } from "@/components/charts/trends"

const RECENT = [
  { id: "JH-2021-ENG-0001", result: "Verified", requester: "ABC Corp", at: "10:12 AM" },
  { id: "JH-2022-ART-0342", result: "Verified", requester: "RecruitCo", at: "09:50 AM" },
  { id: "JH-2020-MSC-9999", result: "Potential Fraud", requester: "XYZ Ltd", at: "09:31 AM" },
]

export default function AdminPage() {
  return (
    <main>
      <SiteHeader />
      <section className="mx-auto max-w-6xl px-4 py-12">
        <h1 className="text-2xl md:text-3xl font-semibold">Super Admin Dashboard</h1>
        <p className="text-muted-foreground mt-2">
          Overview of verification activity, fraud signals, and system health.
        </p>

        <div className="mt-8">
          <SummaryCards />
        </div>

        <div className="mt-8 grid gap-6 md:grid-cols-2">
          <VerificationTrends />
          <div className="rounded-lg border p-4 bg-card">
            <p className="font-medium">Recent Verifications</p>
            <ul className="mt-4 grid gap-2">
              {RECENT.map((r) => (
                <li key={r.id} className="flex items-center justify-between rounded-md border p-3">
                  <div>
                    <p className="text-sm font-medium">{r.id}</p>
                    <p className="text-xs text-muted-foreground">{r.requester}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm">{r.result}</p>
                    <p className="text-xs text-muted-foreground">{r.at}</p>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>
      <SiteFooter />
    </main>
  )
}
