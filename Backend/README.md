# Document Information Extraction & Confidence Scoring Pipeline

## 1. Overview
This project extracts important information from documents (PDFs or images) using **OCR, NER (spaCy), and regex** and calculates **confidence scores** using **fuzzy text matching**.  

Key outputs include:  
- Name, Roll No, Certificate ID  
- CGPA, Date of Issue, Institution Name  
- Confidence score for each field and overall document

---

## 2. Features
- Handles **PDF and image inputs** seamlessly  
- Preprocesses images for better OCR accuracy: binarization, noise removal, inversion  
- Uses **Tesseract OCR** to extract text from images  
- Extracts entities using **spaCy NER** + **regex rules**  
- Calculates **per-field and overall confidence scores** using **fuzzy text matching**  

---

## 3. Dependencies
Install the required libraries:  

```bash
pip install opencv-python
pip install pytesseract
pip install pdf2image
pip install spacy
python -m spacy download en_core_web_sm
pip install fuzzywuzzy
pip install python-Levenshtein 
```

---

## 4. Pipeline Steps

### Step 1: Input Handling
- Detect if input is a **PDF or image** (bool flag `is_pdf`)  
- For PDFs:
  - Convert each page to **JPEG images** using `pdf2image`  
  - Save images in an output directory  
- For images:
  - Use directly for preprocessing  

---

### Step 2: Image Preprocessing
Performed on each image page:
1. **Load Image** – grayscale or color  
2. **Binarization** – Otsu / adaptive thresholding  
3. **Noise Removal** – median filter, opening, or closing  
4. **Morphological Operations** – dilate, erode, open, close  
5. **Optional** – inversion (`cv2.bitwise_not`) if needed for OCR  

*Goal:* Enhance text clarity for better OCR accuracy  

---

### Step 3: OCR (Text Extraction)
- Use **Tesseract OCR via pytesseract**  
- Input: preprocessed image  
- Configurable parameters:
  - `PSM` (Page Segmentation Mode)  
  - `OEM` (OCR Engine Mode)  
- Output: plain text string for further processing  

---

### Step 4: Named Entity Recognition (NER)
- **spaCy NER** for free-text entities:
  - `PERSON` → Name  
  - `ORG/FAC/GPE` → Institution Name  
  - `DATE` → Date of Issue (partial)  
- **Regex rules** for structured fields:
  - Roll No → `Roll No: 123456`  
  - Certificate ID → `Cert ID: CERT-987654`  
  - CGPA → `CGPA: 9.2`  
  - Date of Issue → `Date of Issue: 15/06/2025`  

*Goal:* Extract all key information from OCR text  

---

### Step 5: Fuzzy Text Matching / Confidence Scoring
- Compare extracted entities with **reference/trusted data**  
- Use **`fuzzywuzzy`**:
  - `fuzz.ratio()` for similarity between extracted and reference strings  
- Calculate:
  - **Field-level confidence** (0–100 per entity)  
  - **Overall document confidence** = average of all field scores  

*Goal:* Determine correctness or trustworthiness of extracted info  

---