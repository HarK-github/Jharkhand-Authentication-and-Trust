import { SiteHeader } from "@/components/site-header"
import { SiteFooter } from "@/components/site-footer"
import { VerifyForm } from "@/components/verify-form"

export default function VerifyPage() {
  return (
    <main>
      <SiteHeader />
      <section className="mx-auto max-w-4xl px-4 py-12">
        <h1 className="text-2xl md:text-3xl font-semibold text-balance">Public Verification</h1>
        <p className="mt-2 text-muted-foreground leading-relaxed">
          Upload an image or a document to verify authenticity. New documents will show a blockchain-style progress
          (hashing, anchoring, confirmations). Previously issued/known documents will show AI/ML-based comparisons
          against expected records. This prototype uses mock data and does not persist any information.
        </p>
        <div className="mt-8">
          <VerifyForm />
        </div>
      </section>
      <SiteFooter />
    </main>
  )
}
