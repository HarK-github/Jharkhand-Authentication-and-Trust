import { SiteHeader } from "@/components/site-header"
import { SiteFooter } from "@/components/site-footer"

export default function HelpPage() {
  return (
    <main>
      <SiteHeader />
      <section className="mx-auto max-w-3xl px-4 py-12">
        <h1 className="text-2xl md:text-3xl font-semibold">Help & FAQ</h1>
        <div className="mt-6 grid gap-6">
          <div className="rounded-lg border p-6 bg-card">
            <h2 className="font-medium">What is this?</h2>
            <p className="text-sm text-muted-foreground mt-2">
              A government-backed academic title authenticity prototype for demonstration purposes. No real data is
              stored.
            </p>
          </div>
          <div className="rounded-lg border p-6 bg-card">
            <h2 className="font-medium">How can I verify a certificate?</h2>
            <p className="text-sm text-muted-foreground mt-2">
              Go to the Verify page and enter details or upload a file. The system matches against a mock dataset to
              illustrate the flow.
            </p>
          </div>
          <div className="rounded-lg border p-6 bg-card">
            <h2 className="font-medium">Is my data secure?</h2>
            <p className="text-sm text-muted-foreground mt-2">
              This demo emphasizes privacy and security concepts, but does not persist any data. In production, strict
              RLS, key management, and audit isolation must be enforced.
            </p>
          </div>
        </div>
      </section>
      <SiteFooter />
    </main>
  )
}
