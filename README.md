 
# Jharkhand Academic Trust (JAT) – Certificate Verification System

## Overview

The Jharkhand Academic Trust (JAT) is a blockchain-based certificate verification system designed for academic institutions in Jharkhand. It provides tamper-proof certificate storage using **Hyperledger Fabric** and supports verification of both digital (PDF) and physical (scanned) certificates. AI-assisted modules enhance fraud detection by verifying seals, signatures, text, faces, and fingerprints.

---
 
# Deployed link : https://jharkhand-authentication-and-trust.vercel.app/admin

# Screenshots
<p align="center"> <img src="https://github.com/user-attachments/assets/6e4cf296-f52a-4c30-97fd-269a3b3bca9b" alt="System Architecture" width="700"/> </p> <p align="center"> <img src="https://github.com/user-attachments/assets/e2ebbc98-2529-4b7b-87fe-ef5ac11a97c1" alt="Verification Workflow" width="700"/> </p> <p align="center"> <img src="https://github.com/user-attachments/assets/a1dab024-990c-4b1b-8aa0-2888ccedc987" alt="Detailed Process Flow" width="700"/> </p>

## Project Structure

```bash
Jharkhand-Academic-Trust/
│
├── Backend/                        # AI/Verification logic
│   ├── Fake Degree verification system- AI Layer 2.py
│   ├── FuzzyTextMatching.ipynb
│   ├── NER.ipynb
│   ├── OCR.ipynb
│   ├── face_fingerprint_detection.pynb
│   ├── pdf2image.ipynb
│   ├── pre_processing.ipynb
│   ├── index.js
│   └── README.md
│
├── Frontend/                       # Web application (Next.js)
│   ├── app/
│   ├── components/
│   ├── data/
│   ├── hooks/
│   ├── lib/
│   ├── public/
│   ├── styles/
│   ├── components.json
│   ├── index.js
│   ├── next.config.mjs
│   ├── package.json
│   ├── package-lock.json
│   ├── pnpm-lock.yaml
│   ├── postcss.config.mjs
│   └── tsconfig.json
│
└── Institute_Hyperledger/          # Blockchain layer
    ├── chaincode/certificate-cc/   # Smart contracts
    ├── services/                   # Blockchain services
    ├── contract.js
    ├── docker-compose.yml          # Network setup
    ├── network.sh                  # Hyperledger network scripts
    └── README.md
```

---

## Technology Stack

* **Blockchain**: Hyperledger Fabric, IPFS
* **Backend (AI/Verification)**: Python, PyTorch, OpenCV, spaCy, Tesseract OCR, scikit-learn
* **Frontend**: Next.js (React), TailwindCSS
* **Other Tools**: Docker, Node.js, fuzzywuzzy, DeepFace, fingerprint enhancer

---

## Execution

### Prerequisites

* Docker & Docker Compose
* Node.js (v16 or later)
* Python 3.9+
* Hyperledger Fabric binaries

### Steps

1. **Set up the Hyperledger Network**

   ```bash
   cd Institute_Hyperledger
   ./network.sh up
   ./network.sh createChannel
   ./network.sh deployCC
   ```

2. **Run the Blockchain Services**

   ```bash
   docker-compose up -d
   ```

3. **Backend Setup**

   ```bash
   cd Backend
   python3 -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   pip install -r requirements.txt
   python index.js
   ```

4. **Frontend Setup**

   ```bash
   cd Frontend
   npm install    # or pnpm install
   npm run dev
   ```

5. **Access the Application**
   Open [http://localhost:3000](http://localhost:3000) in a browser to access the verification portal.

---

## Outputs

* Certificate authenticity: Genuine / Suspicious / Fraudulent
* Extracted details: Name, Roll Number, Certificate ID, Institution, CGPA, Date
* Confidence scoring for each field and overall document
* Blockchain verification result via hash or QR comparison

---

## Benefits

* **Students**: Certificates that are secure and universally verifiable
* **Employers**: Quick and reliable candidate verification
* **Universities**: Reduced manual verification and improved credibility
* **Government**: Scalable, trusted framework for academic integrity

---

 
