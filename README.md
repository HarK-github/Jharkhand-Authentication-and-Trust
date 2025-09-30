# Jharkhand-Authentication-and-Trust
Certificate validator for jharkhand acadmeia and institutions.


# Front end
# Backend
 ## Document comparision
# Smart Contract V
A blockchain-based certificate verification solution using Hyperledger Fabric that provides tamper-proof storage of certificate hashes and metadata. Supports both PDF and image-based certificates with AI/ML verification.

## Certificate Issuance Flow

### Certificate Generation
- University ERP system generates certificates (PDF format)
- For physical certificates: Generate with embedded QR code

### Hash Creation & Storage
```
PDF Certificate → SHA-256 Hash → Hyperledger Fabric Blockchain
```
- System computes SHA-256 hash of the PDF certificate
- Writes to blockchain: hash + metadata (student name, roll number, certificate ID, issuing authority, timestamp)

### QR Code Generation (for physical certificates)
- QR code contains: certificate ID + verification URL
- Embedded on physical certificates

## Certificate Verification Flow

### For PDF Certificates
**Direct Hash Comparison**
- User uploads PDF certificate
- System computes SHA-256 hash of uploaded PDF
- Compares with on-chain hash in Hyperledger Fabric
- **Match** → Authentic ✅ | **No Match** → Fake ❌

### For Image Certificates (Photos/Scans)
**QR Code Scanning**
- User uploads image of physical certificate
- System scans QR code to extract certificate ID

## AI/ML Document Processing

### 1. Document Extraction & Confidence Scoring
1. Input + OCR  
   - Handle PDF/Image → Preprocess (binarize, denoise) → OCR with Tesseract  
2. Entity Extraction  
   - spaCy NER (Name, Institution, Date) + Regex (Roll No, Cert ID, CGPA)  
3. Confidence Scoring  
   - Fuzzy matching vs. reference → Field-wise & overall confidence  

---

### 2. Forgery Detection
1. Seal & Stamp Verification  
   - Siamese Network + Contrastive Loss → genuine vs. forged  
2. Signature Verification  
   - Same Siamese model for comparing signatures  
3. Font/Text Analysis  
   - Error Level Analysis (ELA) + OCR text boxes + Random Forest  

---

### 3. Face & Thumbprint Verification
1. Face Recognition  
   - Detect face (MTCNN) → Embedding (DeepFace) → Compare similarity  
2. Thumbprint Matching  
   - Enhance print (fingerprint_enhancer) → Extract minutiae → Compare features  
3. Identity Decision  
   - ✅ Verified if both match  
   - ⚠️ Manual Review if only one matches  
   - ❌ Reject if both mismatch  

---

### ⚙️ Tech Stack
- OCR: Tesseract (pytesseract)  
- NER: spaCy  
- Image Processing: OpenCV, scikit-image  
- Fuzzy Matching: fuzzywuzzy  
- Forgery Detection: PyTorch (Siamese NN), scikit-learn (Random Forest), ELA  
- Biometrics: MTCNN, DeepFace, fingerprint_enhancer  

---

### 🖥️ Outputs
- Extracted fields: Name, Roll No, Cert ID, CGPA, Date, Institution  
- Confidence: field-level + overall  
- Forgery flags: seals, signatures, fonts  
- Biometric verification: face + thumbprint results  
 


**Blockchain Verification**
- Uses certificate ID from QR to fetch original hash from blockchain
- Compares reconstructed data with on-chain record

## Technology Stack

- **Blockchain**: Hyperledger Fabric
- **File Hashing**: SHA-256 for PDF verification
- **Storage**: IPFS for certificate backups

## Key Features

- **PDF Verification**: Direct hash comparison for digital certificates
- **Image Verification**: AI/ML + QR code scanning for physical documents
- **Instant Results**: Real-time authenticity checks
- **Tamper-Proof**: Hyperledger Fabric ensures data integrity
- **Multi-Format**: Supports both digital and physical certificates

## Benefits

- **PDF Certificates**: Fast, direct hash verification
- **Physical Certificates**: AI-powered image processing + QR verification
- **Secure**: Hyperledger Fabric ensures immutability
- **Accurate**: Combined blockchain + AI verification
- **User-Friendly**: Simple upload/scan process
