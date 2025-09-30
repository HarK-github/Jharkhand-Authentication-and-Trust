"use client"

import type React from "react"

import { SiteHeader } from "@/components/site-header"
import { SiteFooter } from "@/components/site-footer"
import { useState } from "react"

const CONNECTED_DBS = [
  { erp: "SAP S/4HANA", database: "PostgreSQL (Neon)", status: "Healthy", lastSync: "2025-09-15 10:22" },
  { erp: "Oracle PeopleSoft", database: "Oracle Autonomous DB", status: "Degraded", lastSync: "2025-09-15 09:47" },
  { erp: "TallyPrime (Education)", database: "MySQL", status: "Healthy", lastSync: "2025-09-14 21:05" },
  {
    erp: "Custom ERP - Ranchi Univ.",
    database: "Supabase (Postgres)",
    status: "Sync paused",
    lastSync: "2025-09-12 13:10",
  },
]

export default function InstitutionPage() {
  const [rows, setRows] = useState("")
  const [submitted, setSubmitted] = useState(false)

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    // Simulate processing without persistence
    setSubmitted(true)
  }

  return (
    <main>
      <SiteHeader />
      <section className="mx-auto max-w-5xl px-4 py-12">
        <h1 className="text-2xl md:text-3xl font-semibold">Institution Portal</h1>
        <p className="text-muted-foreground mt-2">
          Upload new certificate issuances (CSV) and manage your institution’s verification footprint. This demo does
          not store any data.
        </p>

        <div className="mt-8 grid gap-6 md:grid-cols-2">
          <form onSubmit={handleSubmit} className="rounded-lg border p-6 bg-card grid gap-4">
            <div>
              <label htmlFor="csv" className="text-sm font-medium">
                CSV Upload (simulated)
              </label>
              <textarea
                id="csv"
                rows={8}
                placeholder={`certificateId,fullName,institution,program,graduationYear,issuedAt,signatureHash
JH-2024-ENG-0101,Sunita Devi,Ranchi University,B.Tech (Civil),2024,2024-06-15T00:00:00.000Z,sig_xxx...`}
                value={rows}
                onChange={(e) => setRows(e.target.value)}
                className="mt-1 w-full rounded-md border bg-background px-3 py-2 text-sm"
              />
              <p className="text-xs text-muted-foreground mt-1">Paste CSV rows. Data is not persisted in this demo.</p>
            </div>
            <button
              type="submit"
              className="inline-flex items-center rounded-md bg-primary text-primary-foreground px-4 py-2 text-sm"
            >
              Submit Batch
            </button>
            {submitted && (
              <div className="rounded-md border p-3 bg-background">
                <p className="text-sm">Batch accepted (simulated). 0 records stored in this demo.</p>
              </div>
            )}
          </form>

          <div className="rounded-lg border p-6 bg-card">
            <p className="font-medium">Guidelines</p>
            <ul className="mt-3 text-sm text-muted-foreground list-disc pl-5 leading-relaxed">
              <li>Ensure certificate IDs are unique and signed by your issuer key.</li>
              <li>Only include necessary fields—avoid personal sensitive information.</li>
              <li>Keep your issuer keys secure and rotate periodically.</li>
            </ul>
          </div>
        </div>

        <div className="mt-8 rounded-lg border p-6 bg-card">
          <p className="font-medium">Connected Databases (ERP)</p>
          <p className="text-xs text-muted-foreground mt-1">
            Simulated view of databases connected with your institute’s ERP.
          </p>
          <div className="mt-4 grid gap-3">
            {CONNECTED_DBS.map((row) => (
              <div
                key={`${row.erp}-${row.database}`}
                className="grid md:grid-cols-4 gap-2 rounded-md border p-3 bg-background"
              >
                <div>
                  <p className="text-sm font-medium">ERP</p>
                  <p className="text-sm text-muted-foreground">{row.erp}</p>
                </div>
                <div>
                  <p className="text-sm font-medium">Database</p>
                  <p className="text-sm text-muted-foreground">{row.database}</p>
                </div>
                <div>
                  <p className="text-sm font-medium">Status</p>
                  <p className="text-sm text-muted-foreground">{row.status}</p>
                </div>
                <div>
                  <p className="text-sm font-medium">Last Sync</p>
                  <p className="text-sm text-muted-foreground">{row.lastSync}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
      <SiteFooter />
    </main>
  )
}
