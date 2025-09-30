import Link from "next/link"

export function Hero() {
  return (
    <section className="bg-secondary">
      <div className="mx-auto max-w-6xl px-4 py-16 flex flex-col md:flex-row items-center gap-10">
        <div className="flex-1">
          <h1 className="text-3xl md:text-5xl font-semibold text-balance">Verify Academic Titles with Confidence</h1>
          <p className="mt-4 text-muted-foreground leading-relaxed">
            A government-backed verification system to validate academic certificates, protect employers from fraud, and
            streamline institutional compliance.
          </p>
          <div className="mt-6 flex items-center gap-3">
            <Link
              href="/verify"
              className="inline-flex items-center justify-center rounded-md bg-primary text-primary-foreground px-4 py-2 text-sm"
            >
              Verify a Certificate
            </Link>
            <Link
              href="/institution"
              className="inline-flex items-center justify-center rounded-md border px-4 py-2 text-sm"
            >
              Institution Portal
            </Link>
          </div>
          <div className="mt-6 grid gap-2 text-sm text-muted-foreground">
            <div>• Tamper-evident logs</div>
            <div>• Instant verification results</div>
            <div>• Privacy-first by design</div>
          </div>
        </div>
        <div className="flex-1">
          <img
            src="/india-government-certificate-verification-illustra.jpg"
            alt="Certificate verification illustration for Indian institutions"
            className="w-full rounded-lg border"
          />
        </div>
      </div>
    </section>
  )
}
