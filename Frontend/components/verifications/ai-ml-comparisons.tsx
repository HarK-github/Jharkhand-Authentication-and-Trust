import { Progress } from "@/components/ui/progress"

type Metrics = {
  nameMatch: number
  institutionMatch: number
  programMatch: number
  signatureIntegrity: number
  layoutSimilarity: number
  modelScores: Array<{ model: string; score: number }>
}

export default function AiMlComparisons({
  extracted,
  metrics,
}: {
  extracted: {
    certificateId?: string
    fullName?: string
    institution?: string
    program?: string
    graduationYear?: number
  }
  metrics: Metrics
}) {
  return (
    <div className="grid gap-4">
      <div>
        <p className="text-sm font-medium">AI/ML Comparisons</p>
        <p className="text-xs text-muted-foreground">
          OCR and model-assisted similarity checks against expected issuance records.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        <div className="rounded-lg border p-4 bg-card">
          <p className="text-sm font-medium">Extracted Details</p>
          <div className="mt-2 grid gap-1 text-sm">
            <p>
              <span className="font-medium">Certificate ID:</span> {extracted.certificateId ?? "—"}
            </p>
            <p>
              <span className="font-medium">Name:</span> {extracted.fullName ?? "—"}
            </p>
            <p>
              <span className="font-medium">Institution:</span> {extracted.institution ?? "—"}
            </p>
            <p>
              <span className="font-medium">Program:</span> {extracted.program ?? "—"}
            </p>
            <p>
              <span className="font-medium">Graduation Year:</span> {extracted.graduationYear ?? "—"}
            </p>
          </div>
        </div>

        <div className="rounded-lg border p-4 bg-card">
          <p className="text-sm font-medium">Similarity Scores</p>
          <div className="mt-3 grid gap-3">
            <Metric label="Name Match" value={metrics.nameMatch} />
            <Metric label="Institution Match" value={metrics.institutionMatch} />
            <Metric label="Program Match" value={metrics.programMatch} />
            <Metric label="Signature Integrity" value={metrics.signatureIntegrity} />
            <Metric label="Layout Similarity" value={metrics.layoutSimilarity} />
          </div>
        </div>
      </div>

      <div className="rounded-lg border p-4 bg-card">
        <p className="text-sm font-medium">Model Outputs</p>
        <div className="mt-3 grid gap-2">
          {metrics.modelScores.map((m) => (
            <div key={m.model} className="grid gap-1">
              <div className="flex items-center justify-between">
                <p className="text-sm">{m.model}</p>
                <p className="text-xs text-muted-foreground">{Math.round(m.score * 100)}%</p>
              </div>
              <Progress value={m.score * 100} aria-label={`${m.model} score`} />
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: number }) {
  const pct = Math.round(value * 100)
  return (
    <div>
      <div className="flex items-center justify-between">
        <p className="text-sm">{label}</p>
        <p className="text-xs text-muted-foreground">{pct}%</p>
      </div>
      <Progress value={pct} aria-label={`${label} similarity`} />
    </div>
  )
}
