import { Progress } from "@/components/ui/progress"

type Props = {
  progress: number
  step: number
  steps: string[]
}

export default function BlockchainProgress({ progress, step, steps }: Props) {
  return (
    <div className="grid gap-4">
      <div className="flex items-center gap-3 overflow-x-auto">
        {steps.map((label, idx) => (
          <div key={label} className="flex items-center gap-3">
            <div
              className={[
                "rounded-md border px-3 py-2 bg-card",
                idx < step ? "opacity-100" : idx === step ? "opacity-100 ring-1 ring-primary" : "opacity-60",
              ].join(" ")}
              aria-current={idx === step ? "step" : undefined}
            >
              <div className="text-xs font-medium">{label}</div>
            </div>
            {idx < steps.length - 1 && <div aria-hidden className="h-px w-8 bg-border" />}
          </div>
        ))}
      </div>

      <div>
        <div className="flex items-center justify-between">
          <p className="text-sm font-medium">Anchoring Progress</p>
          <p className="text-xs text-muted-foreground">{Math.floor(progress)}%</p>
        </div>
        <Progress className="mt-2" value={progress} aria-label="Blockchain anchoring progress" />
        <p className="text-xs text-muted-foreground mt-2">
          Process: hashing file contents, anchoring hash on chain, and confirmations.
        </p>
      </div>
    </div>
  )
}
