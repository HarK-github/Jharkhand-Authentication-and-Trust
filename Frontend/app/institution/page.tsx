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
          Upload new certificate issuances (CSV) and manage your institution’s verification footprint.
        </p>

        <div className="mt-8 grid gap-6 md:grid-cols-2">
          <form onSubmit={handleSubmit} className="rounded-lg border p-6 bg-card grid gap-4">
            <div>
              <label htmlFor="csv" className="text-sm font-medium">
                CSV Upload
              </label>
              <textarea
                id="csv"
                rows={8}
                placeholder={`certificate_id,full_name,institution,program,graduation_year,issued_at,signature_hash
JH-2024-ENG-0101,Sunita Devi,Ranchi University,B.Tech (Civil),2024,2024-06-15T00:00:00.000Z,sig_xxx...`}
                value={rows}
                onChange={(e) => setRows(e.target.value)}
                className="mt-1 w-full rounded-md border bg-background px-3 py-2 text-sm"
              />
              <p className="text-xs text-muted-foreground mt-1">Paste CSV rows. This is for illustration purposes.</p>
            </div>
            <button
              type="submit"
              className="inline-flex items-center rounded-md bg-primary text-primary-foreground px-4 py-2 text-sm"
            >
              Submit Batch
            </button>
            {submitted && (
              <div className="rounded-md border p-3 bg-background">
                <p className="text-sm">Batch received.</p>
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
          <p className="text-xs text-muted-foreground mt-1">View of databases connected with your institute’s ERP.</p>
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

        <div className="mt-8 rounded-lg border p-6 bg-card">
          <p className="font-medium">SQL: Linking ERP Database to JAT Central Database</p>
          <p className="text-xs text-muted-foreground mt-1">Example schemas and joins for integration.</p>
          <pre className="mt-3 whitespace-pre-wrap text-xs bg-background rounded-md border p-3 overflow-x-auto">
            <code>{`-- Central JAT database (PostgreSQL)
CREATE TABLE institutions (
  id UUID PRIMARY KEY,
  name TEXT NOT NULL,
  erp_name TEXT,
  issuer_public_key TEXT NOT NULL
);

CREATE TABLE certificates (
  certificate_id TEXT PRIMARY KEY,
  institution_id UUID REFERENCES institutions(id),
  full_name TEXT NOT NULL,
  program TEXT NOT NULL,
  graduation_year INTEGER NOT NULL,
  issued_at TIMESTAMPTZ NOT NULL,
  signature_hash TEXT NOT NULL
);

CREATE TABLE certificate_anchors (
  certificate_id TEXT REFERENCES certificates(certificate_id),
  anchor_tx_id TEXT NOT NULL,
  anchored_at TIMESTAMPTZ NOT NULL,
  confirmations INTEGER DEFAULT 0,
  PRIMARY KEY (certificate_id, anchor_tx_id)
);

-- Example ERP schema (institute side)
-- Assume ERP schema name: erp
CREATE TABLE erp.students (
  student_id TEXT PRIMARY KEY,
  full_name TEXT NOT NULL,
  institution TEXT NOT NULL,
  program TEXT NOT NULL,
  graduation_year INTEGER NOT NULL
);

CREATE TABLE erp.issuances (
  certificate_id TEXT PRIMARY KEY,
  student_id TEXT REFERENCES erp.students(student_id),
  issued_at TIMESTAMPTZ NOT NULL,
  signature_hash TEXT NOT NULL
);

-- Linking query (read-only join)
SELECT
  i.name AS institution,
  c.certificate_id,
  c.full_name,
  c.program,
  c.graduation_year,
  c.issued_at,
  a.anchor_tx_id,
  a.confirmations
FROM certificates c
JOIN institutions i ON i.id = c.institution_id
LEFT JOIN certificate_anchors a ON a.certificate_id = c.certificate_id
WHERE c.certificate_id = 'JH-2024-ENG-0101';

-- Example: materialized view for fast comparisons
CREATE MATERIALIZED VIEW mv_certificate_index AS
SELECT
  c.certificate_id,
  LOWER(c.full_name) AS full_name_idx,
  LOWER(c.program) AS program_idx,
  c.graduation_year,
  c.institution_id
FROM certificates c;

-- Example: ingest from ERP to central (nightly)
INSERT INTO certificates (certificate_id, institution_id, full_name, program, graduation_year, issued_at, signature_hash)
SELECT
  e.certificate_id,
  i.id AS institution_id,
  s.full_name,
  s.program,
  s.graduation_year,
  e.issued_at,
  e.signature_hash
FROM erp.issuances e
JOIN erp.students s ON s.student_id = e.student_id
JOIN institutions i ON i.name = s.institution
ON CONFLICT (certificate_id) DO NOTHING;`}</code>
          </pre>
        </div>
      </section>
      <SiteFooter />
    </main>
  )
}
