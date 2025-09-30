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

### Document Extraction & Forgery Detection – Summary

#### 🔄 Pipeline
1. Input Handling  
   - Detect PDF / Image  
   - Convert PDF → Images  

2. Preprocessing  
   - Grayscale, Binarization  
   - Noise removal, Morphology  
   - Inversion (if needed)  

3. OCR (Tesseract)  
   - Extract raw text  
   - Configurable PSM & OEM  

4. Entity Extraction  
   - spaCy NER → Name, Institution, Date  
   - Regex → Roll No, Cert ID, CGPA, Date of Issue  

5. Confidence Scoring (Fuzzy Matching)  
   - Field-level score (0–100)  
   - Overall confidence = average of all fields  

---

#### 🔍 Forgery Detection
- Seal & Stamp Verification → Siamese Network + Contrastive Loss  
- Signature Verification → Same Siamese model  
- Font/Text Analysis  
  - Error Level Analysis (ELA)  
  - OCR text boxes + font features  
  - Random Forest classifier → genuine vs. forged  

---

#### ⚙️ Tech Stack
- OCR: Tesseract (pytesseract)  
- NER: spaCy  
- Preprocessing: OpenCV  
- Fuzzy Matching: fuzzywuzzy  
- Forgery Detection: PyTorch (Siamese NN), scikit-learn (Random Forest), ELA  

---

#### 🖥️ Example Outputs
- Extracted fields: Name, Roll No, Cert ID, CGPA, Date, Institution  
- Confidence: per field + overall  
- Forgery results: seals, signatures, fonts  


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
