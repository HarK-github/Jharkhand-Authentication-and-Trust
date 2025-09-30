export type CertificateRecord = {
  certificateId: string
  fullName: string
  institution: string
  program: string
  graduationYear: number
  issuedAt: string
  signatureHash: string
}

export const MOCK_CERTIFICATES: CertificateRecord[] = [
  {
    certificateId: "JH-2021-ENG-0001",
    fullName: "Amit Kumar",
    institution: "Ranchi University",
    program: "B.Tech (Mechanical Engineering)",
    graduationYear: 2021,
    issuedAt: "2021-06-30T00:00:00.000Z",
    signatureHash: "sig_9f8a1c2b3d4e",
  },
  {
    certificateId: "JH-2022-ART-0342",
    fullName: "Priya Singh",
    institution: "Kolhan University",
    program: "BA (Economics)",
    graduationYear: 2022,
    issuedAt: "2022-08-15T00:00:00.000Z",
    signatureHash: "sig_a1b2c3d4e5f6",
  },
  {
    certificateId: "JH-2020-MSC-1123",
    fullName: "Rahul Verma",
    institution: "BIT Sindri",
    program: "M.Sc (Computer Science)",
    graduationYear: 2020,
    issuedAt: "2020-05-10T00:00:00.000Z",
    signatureHash: "sig_abc123def456",
  },
]
