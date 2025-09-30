import { SiteHeader } from "@/components/site-header"
import { SiteFooter } from "@/components/site-footer"
import { Hero } from "@/components/hero"

export default function Page() {
  return (
    <main>
      <SiteHeader />
      <Hero />
      <section className="mx-auto max-w-6xl px-4 py-12">
        <h2 className="text-xl font-semibold">Trusted by Employers and Institutions</h2>
        <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-6">
          {["Public Sector Units", "Universities", "Recruitment Agencies", "Enterprises"].map((label) => (
            <div key={label} className="rounded-lg border p-4 text-center text-sm">
              {label}
            </div>
          ))}
        </div>
      </section>
      <section className="mx-auto max-w-6xl px-4 pb-16">
        <div className="grid gap-6 md:grid-cols-3">
          <div className="rounded-lg border p-6">
            <h3 className="font-medium">Fast Verification</h3>
            <p className="text-sm text-muted-foreground mt-2">
              Enter certificate details or upload a file to verify in seconds.
            </p>
          </div>
          <div className="rounded-lg border p-6">
            <h3 className="font-medium">Secure & Auditable</h3>
            <p className="text-sm text-muted-foreground mt-2">
              Every check produces a tamper-evident record for compliance and audits.
            </p>
          </div>
          <div className="rounded-lg border p-6">
            <h3 className="font-medium">Privacy-First</h3>
            <p className="text-sm text-muted-foreground mt-2">
              Only necessary fields are processed. No persistent storage in this prototype.
            </p>
          </div>
        </div>
      </section>
      <SiteFooter />
    </main>
  )
}
