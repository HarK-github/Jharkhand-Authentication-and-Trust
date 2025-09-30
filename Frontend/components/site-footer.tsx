export function SiteFooter() {
  return (
    <footer className="border-t bg-background">
      <div className="mx-auto max-w-6xl px-4 py-8 grid gap-4 md:grid-cols-3">
        <div>
          <p className="font-medium">Government of Jharkhand</p>
          <p className="text-sm text-muted-foreground">Academic Title Authenticity Initiative</p>
        </div>
        <div>
          <p className="font-medium">Security & Privacy</p>
          <ul className="text-sm text-muted-foreground list-disc pl-5">
            <li>Data minimization and purpose limitation</li>
            <li>Tamper-evident audit trail</li>
            <li>Strong cryptographic signatures</li>
          </ul>
        </div>
        <div>
          <p className="font-medium">Contact</p>
          <p className="text-sm text-muted-foreground">support@example.gov.in</p>
        </div>
      </div>
      <div className="mx-auto max-w-6xl px-4 pb-6 text-xs text-muted-foreground">
        © {new Date().getFullYear()} Govt. of Jharkhand. Demonstration prototype only.
      </div>
    </footer>
  )
}
