import { SiteHeader } from "@/components/site-header"
import { SiteFooter } from "@/components/site-footer"
import { SummaryCards } from "@/components/admin/summary-cards"
import { VerificationTrends } from "@/components/charts/trends"
import { VerificationBreakdown } from "@/components/charts/circular"

const RECENT = [
  { id: "JH-2021-ENG-0001", result: "Verified", requester: "Tata Consultancy Services", at: "10:12 AM" },
  { id: "JH-2022-ART-0342", result: "Verified", requester: "Infosys Ltd", at: "09:50 AM" },
  { id: "JH-2020-MSC-9999", result: "Potential Fraud", requester: "Wipro", at: "09:31 AM" },
]

export default function AdminPage() {
  return (
    <main>
      <SiteHeader />
      <section className="mx-auto max-w-6xl px-4 py-12">
        <h1 className="text-2xl md:text-3xl font-semibold">Analysis Dashboard</h1>
        <p className="text-muted-foreground mt-2">
          Overview of verification activity, fraud signals, and system health.
        </p>

        <div className="mt-8">
          <SummaryCards />
        </div>

        <div className="mt-8 grid gap-6 md:grid-cols-2">
          <VerificationTrends />
          <VerificationBreakdown />
        </div>
      </section>
      <SiteFooter />
    </main>
  )
}
