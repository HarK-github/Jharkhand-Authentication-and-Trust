import { NextResponse } from "next/server"
import { MOCK_CERTIFICATES } from "@/data/mock-certificates"

export async function POST(req: Request) {
  const body = await req.json()
  const { certificateId, fullName, institution, graduationYear } = body || {}

  const record = MOCK_CERTIFICATES.find((c) => {
    if (certificateId && c.certificateId !== certificateId) return false
    if (fullName && c.fullName.toLowerCase() !== String(fullName).toLowerCase()) return false
    if (institution && c.institution.toLowerCase() !== String(institution).toLowerCase()) return false
    if (graduationYear && c.graduationYear !== Number(graduationYear)) return false
    return true
  })

  if (!record) {
    return NextResponse.json(
      { status: "potential_fraud", reason: "No matching record found", checkedAt: new Date().toISOString() },
      { status: 200 },
    )
  }

  return NextResponse.json({
    status: "verified",
    record,
    checkedAt: new Date().toISOString(),
    attestations: {
      signatureValid: true,
      issuer: record.institution,
    },
  })
}
