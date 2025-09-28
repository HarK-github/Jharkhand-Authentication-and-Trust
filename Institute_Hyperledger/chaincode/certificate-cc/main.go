package main

import "github.com/hyperledger/fabric-contract-api-go/contractapi"

// CertificateContract handles certificate operations
type CertificateContract struct {
    contractapi.Contract
}

// InitLedger initializes the ledger with sample certificates
func (cc *CertificateContract) InitLedger(ctx contractapi.TransactionContextInterface) error

// IssueCertificate creates a new certificate on the ledger
func (cc *CertificateContract) IssueCertificate(ctx contractapi.TransactionContextInterface, 
    certID string, holderName string, course string, issuer string, issueDate string, metadata string) error

// VerifyCertificate checks certificate authenticity and validity
func (cc *CertificateContract) VerifyCertificate(ctx contractapi.TransactionContextInterface, 
    certID string) (bool, error)

// GetCertificate retrieves certificate details by ID
func (cc *CertificateContract) GetCertificate(ctx contractapi.TransactionContextInterface, 
    certID string) (*Certificate, error)

// UpdateCertificateStatus revokes or updates certificate status
func (cc *CertificateContract) UpdateCertificateStatus(ctx contractapi.TransactionContextInterface, 
    certID string, status string) error

// GetCertificateHistory returns the complete history of a certificate
func (cc *CertificateContract) GetCertificateHistory(ctx contractapi.TransactionContextInterface, 
    certID string) ([]HistoryEntry, error)