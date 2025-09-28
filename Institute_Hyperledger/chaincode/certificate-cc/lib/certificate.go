package lib

// Certificate represents the digital certificate structure
type Certificate struct {
    CertID      string `json:"certID"`
    HolderName  string `json:"holderName"`
    Course      string `json:"course"`
    Issuer      string `json:"issuer"`
    IssueDate   string `json:"issueDate"`
    Status      string `json:"status"` // ACTIVE, REVOKED, EXPIRED
    Metadata    string `json:"metadata"` // Additional data or hash
    CreatedAt   string `json:"createdAt"`
}

// HistoryEntry tracks certificate lifecycle changes
type HistoryEntry struct {
    TxID      string      `json:"txId"`
    Timestamp string      `json:"timestamp"`
    IsDelete  bool        `json:"isDelete"`
    Value     Certificate `json:"value"`
}