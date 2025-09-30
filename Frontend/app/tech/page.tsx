import { SiteHeader } from "@/components/site-header"
import { SiteFooter } from "@/components/site-footer"

export default function TechPage() {
  return (
    <main>
      <SiteHeader />
      <section className="mx-auto max-w-4xl px-4 py-12">
        <h1 className="text-2xl md:text-3xl font-semibold">Technology & Security</h1>
        <div className="mt-6 grid gap-6">
          <div className="rounded-lg border p-6 bg-card">
            <h2 className="font-medium">Cryptographic Signatures</h2>
            <p className="text-sm text-muted-foreground mt-2">
              Certificates are signed by issuing institutions. Verification recomputes and checks signature integrity
              against trusted issuer keys. This prototype simulates signatures.
            </p>
          </div>
          <div className="rounded-lg border p-6 bg-card">
            <h2 className="font-medium">QR Verification Flow</h2>
            <p className="text-sm text-muted-foreground mt-2">
              Graduates can embed a QR code on their documents. Scanning opens the public verification page with the
              certificate ID pre-filled for instant checks.
            </p>
          </div>
          <div className="rounded-lg border p-6 bg-card">
            <h2 className="font-medium">Tamper-Evident Audit Trail</h2>
            <p className="text-sm text-muted-foreground mt-2">
              Each verification generates a record that can be anchored to a tamper-evident log. This demo does not
              persist data but illustrates the UX and flows.
            </p>
          </div>
        </div>
      </section>
      <SiteFooter />
    </main>
  )
}
