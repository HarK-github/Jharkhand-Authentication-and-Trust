"use client"

import type React from "react"
import { useEffect, useRef, useState } from "react"

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import AiMlComparisons from "./verifications/ai-ml-comparisons"
import BlockchainProgress from "./verifications/blockchain-progress"

type Result =
  | { status: "idle" | "loading" }
  | {
      status: "verified" | "potential_fraud"
      checkedAt: string
      record?: {
        certificateId: string
        fullName: string
        institution: string
        program: string
        graduationYear: number
        issuedAt: string
        signatureHash: string
      }
      reason?: string
    }

type UploadFlow =
  | { type: "none" }
  | { type: "progress"; step: number; progress: number }
  | {
      type: "old"
      // simple metrics for demo purposes
      metrics: {
        nameMatch: number
        institutionMatch: number
        programMatch: number
        signatureIntegrity: number
        layoutSimilarity: number
        modelScores: Array<{ model: string; score: number }>
      }
      extracted: {
        certificateId?: string
        fullName?: string
        institution?: string
        program?: string
        graduationYear?: number
      }
    }
  | {
      type: "new"
      anchorTxId: string
      confirmations: number
    }

export function VerifyForm() {
  const [certificateId, setCertificateId] = useState("")
  const [fullName, setFullName] = useState("")
  const [institution, setInstitution] = useState("")
  const [graduationYear, setGraduationYear] = useState<number | "">("")
  const [result, setResult] = useState<Result>({ status: "idle" })

  const [tab, setTab] = useState<"image" | "document">("image")
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [docFile, setDocFile] = useState<File | null>(null)
  const [uploadFlow, setUploadFlow] = useState<UploadFlow>({ type: "none" })
  const intervalRef = useRef<number | null>(null)

  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        window.clearInterval(intervalRef.current)
      }
    }
  }, [])

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    setUploadFlow({ type: "none" })
    setResult({ status: "loading" })
    try {
      const res = await fetch("/api/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          certificateId: certificateId || undefined,
          fullName: fullName || undefined,
          institution: institution || undefined,
          graduationYear: graduationYear || undefined,
        }),
      })
      const data = await res.json()
      setResult(data)
    } catch {
      setResult({ status: "potential_fraud", checkedAt: new Date().toISOString(), reason: "Network error" })
    }
  }

  function extractFromFileName(name: string) {
    const lower = name.toLowerCase()
    const isOld = lower.includes("jh-") || lower.includes("known") || lower.includes("old") || lower.includes("amit")
    const maybeIdMatch = name.match(/JH-\d{4}-[A-Z]{3}-\d{4}/)
    const certificateIdGuess = maybeIdMatch?.[0]
    return { isOld, certificateIdGuess }
  }

  function handleVerifyUpload() {
    setResult({ status: "idle" })
    const file = tab === "image" ? imageFile : docFile
    if (!file) return

    const { isOld, certificateIdGuess } = extractFromFileName(file.name)

    if (isOld) {
      // Show AI/ML comparisons for an "old" (known) document
      setUploadFlow({
        type: "old",
        extracted: {
          certificateId: certificateIdGuess ?? "JH-2021-ENG-0001",
          fullName: "Amit Kumar",
          institution: "Ranchi University",
          program: "B.Tech (Mechanical)",
          graduationYear: 2021,
        },
        metrics: {
          nameMatch: 0.98,
          institutionMatch: 1.0,
          programMatch: 0.97,
          signatureIntegrity: 1.0,
          layoutSimilarity: 0.95,
          modelScores: [
            { model: "OCR+FuzzyName", score: 0.98 },
            { model: "LayoutNet v2", score: 0.95 },
            { model: "SigVerify SHA-256", score: 1.0 },
          ],
        },
      })
    } else {
      // New document: show blockchain-style progress (hashing, anchoring, confirmations)
      let progress = 0
      setUploadFlow({ type: "progress", step: 0, progress })
      if (intervalRef.current) window.clearInterval(intervalRef.current)
      intervalRef.current = window.setInterval(() => {
        progress = Math.min(progress + 8, 100)
        const step = progress < 25 ? 0 : progress < 50 ? 1 : progress < 85 ? 2 : 3
        setUploadFlow({ type: "progress", step, progress })
        if (progress >= 100 && intervalRef.current) {
          window.clearInterval(intervalRef.current)
          setUploadFlow({
            type: "new",
            anchorTxId: "0x" + Math.random().toString(16).slice(2, 10) + "…",
            confirmations: 12,
          })
        }
      }, 300)
    }
  }

  return (
    <div className="grid gap-6">
      {/* Upload options */}
      <div className="rounded-lg border p-6 bg-card grid gap-4">
        <div>
          <p className="text-sm font-medium">Upload to Verify</p>
          <p className="text-xs text-muted-foreground mt-1">
            Choose an image or a document. New documents will simulate blockchain anchoring. Known documents will show
            AI/ML comparisons.
          </p>
        </div>
        <Tabs value={tab} onValueChange={(v) => setTab(v as "image" | "document")}>
          <TabsList className="grid grid-cols-2">
            <TabsTrigger value="image">Image</TabsTrigger>
            <TabsTrigger value="document">Document</TabsTrigger>
          </TabsList>
          <TabsContent value="image" className="mt-4">
            <input
              type="file"
              accept="image/*"
              onChange={(e) => setImageFile(e.target.files?.[0] ?? null)}
              className="block w-full text-sm"
              aria-label="Upload certificate image"
            />
            {imageFile && <p className="text-xs text-muted-foreground mt-2">Selected: {imageFile.name}</p>}
          </TabsContent>
          <TabsContent value="document" className="mt-4">
            <input
              type="file"
              accept=".pdf,.doc,.docx"
              onChange={(e) => setDocFile(e.target.files?.[0] ?? null)}
              className="block w-full text-sm"
              aria-label="Upload certificate document"
            />
            {docFile && <p className="text-xs text-muted-foreground mt-2">Selected: {docFile.name}</p>}
          </TabsContent>
        </Tabs>
        <div className="flex gap-3">
          <button
            type="button"
            onClick={handleVerifyUpload}
            className="inline-flex items-center rounded-md bg-primary text-primary-foreground px-4 py-2 text-sm"
          >
            Verify Upload
          </button>
          <button
            type="button"
            onClick={() => {
              setImageFile(null)
              setDocFile(null)
              setUploadFlow({ type: "none" })
            }}
            className="inline-flex items-center rounded-md border px-4 py-2 text-sm"
          >
            Reset
          </button>
        </div>

        {/* Upload result areas */}
        {uploadFlow.type === "progress" && (
          <div className="rounded-lg border p-4 bg-background">
            <BlockchainProgress
              progress={uploadFlow.progress}
              step={uploadFlow.step}
              steps={["Parse & OCR", "Compute Hash", "Anchor on Chain", "Confirmations"]}
            />
          </div>
        )}

        {uploadFlow.type === "new" && (
          <div className="rounded-lg border p-4 bg-background grid gap-2">
            <p className="text-sm font-medium">Anchoring Complete</p>
            <p className="text-xs text-muted-foreground">
              Anchored TX: {uploadFlow.anchorTxId} • Confirmations: {uploadFlow.confirmations}
            </p>
          </div>
        )}

        {uploadFlow.type === "old" && (
          <div className="rounded-lg border p-4 bg-background">
            <AiMlComparisons extracted={uploadFlow.extracted} metrics={uploadFlow.metrics} />
          </div>
        )}
      </div>

      {/* Manual entry (kept for completeness) */}
      <form onSubmit={onSubmit} className="rounded-lg border p-6 grid gap-4 bg-card">
        <p className="text-sm font-medium">Or verify via details</p>
        <div>
          <label htmlFor="certificateId" className="text-sm font-medium">
            Certificate ID
          </label>
          <input
            id="certificateId"
            value={certificateId}
            onChange={(e) => setCertificateId(e.target.value)}
            placeholder="e.g., JH-2021-ENG-0001"
            className="mt-1 w-full rounded-md border bg-background px-3 py-2 text-sm"
          />
        </div>
        <div className="grid md:grid-cols-3 gap-4">
          <div>
            <label htmlFor="fullName" className="text-sm font-medium">
              Full Name
            </label>
            <input
              id="fullName"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              className="mt-1 w-full rounded-md border bg-background px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label htmlFor="institution" className="text-sm font-medium">
              Institution
            </label>
            <input
              id="institution"
              value={institution}
              onChange={(e) => setInstitution(e.target.value)}
              className="mt-1 w-full rounded-md border bg-background px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label htmlFor="graduationYear" className="text-sm font-medium">
              Graduation Year
            </label>
            <input
              id="graduationYear"
              inputMode="numeric"
              pattern="[0-9]*"
              value={graduationYear}
              onChange={(e) => setGraduationYear(e.target.value ? Number(e.target.value) : "")}
              className="mt-1 w-full rounded-md border bg-background px-3 py-2 text-sm"
            />
          </div>
        </div>
        <div className="flex gap-3">
          <button
            type="submit"
            className="inline-flex items-center rounded-md bg-primary text-primary-foreground px-4 py-2 text-sm"
          >
            Verify Now
          </button>
          <button
            type="button"
            onClick={() => {
              setCertificateId("JH-2021-ENG-0001")
              setFullName("Amit Kumar")
              setInstitution("Ranchi University")
              setGraduationYear(2021)
            }}
            className="inline-flex items-center rounded-md border px-4 py-2 text-sm"
          >
            Prefill Example
          </button>
        </div>
      </form>

      {result.status !== "idle" && result.status !== "loading" && (
        <div className="rounded-lg border p-6 bg-card">
          <h3 className="font-semibold">Result</h3>
          {result.status === "verified" ? (
            <div className="mt-2 grid gap-2">
              <p className="text-sm">Status: Verified</p>
              <p className="text-sm text-muted-foreground">Checked at: {new Date(result.checkedAt).toLocaleString()}</p>
              <div className="mt-2 rounded-md border p-4">
                <p className="text-sm">
                  <span className="font-medium">Certificate ID:</span> {result.record?.certificateId}
                </p>
                <p className="text-sm">
                  <span className="font-medium">Name:</span> {result.record?.fullName}
                </p>
                <p className="text-sm">
                  <span className="font-medium">Institution:</span> {result.record?.institution}
                </p>
                <p className="text-sm">
                  <span className="font-medium">Program:</span> {result.record?.program}
                </p>
                <p className="text-sm">
                  <span className="font-medium">Graduation Year:</span> {result.record?.graduationYear}
                </p>
              </div>
            </div>
          ) : (
            <div className="mt-2 grid gap-2">
              <p className="text-sm">Status: Potential Fraud</p>
              <p className="text-sm text-muted-foreground">
                {result.reason || "The provided details did not match our records."}
              </p>
            </div>
          )}
        </div>
      )}

      {result.status === "loading" && (
        <div className="rounded-lg border p-6 bg-card">
          <p className="text-sm">Verifying…</p>
          <Progress value={35} className="mt-2" aria-label="Verifying progress" />
        </div>
      )}
    </div>
  )
}
