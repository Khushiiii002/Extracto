# extracto — Invoice Intelligence

Parse invoices in seconds using Vision AI. Upload a file and receive structured, clean JSON output instantly.

## Live Demo

Access the application here:  
https://extracto-production.up.railway.app/

---

## Overview

extracto is a vision-powered invoice extraction system that converts unstructured invoice documents into structured, machine-readable data.

It supports images and PDFs and is designed for fast, reliable extraction of key billing information without manual intervention.

---

## Key Features

- Fast processing with results typically under five seconds  
- Vision-based extraction using Qwen2-VL-2B Instruct  
- Supports JPG, PNG, and PDF files up to 10MB  
- Multi-currency detection including USD, EUR, GBP, INR, JPY, and AED  
- Automatic retry mechanism with exponential backoff for reliability  
- Input validation with file type and size enforcement  
- Deduplication using SHA-256 hashing to prevent redundant processing  

---

## Extracted Fields

| Field | Description |
|------|-------------|
| Invoice Number | Unique invoice identifier |
| Date | Invoice issue date |
| Vendor Name | Supplier or issuing company |
| Customer Name | Recipient of the invoice |
| Total Amount | Final payable amount |
| Line Items | Item-level breakdown with quantity, unit price, and totals |
| Notes | Payment terms or additional remarks |

---

## Tech Stack

| Layer | Technology |
|------|------------|
| Backend API | FastAPI (Python 3.11) |
| AI Model | Qwen2-VL-2B Instruct |
| Inference GPU | Lightning AI (A10G) |
| Deployment | Railway (Docker-based) |
| Core Libraries | PyTorch, Transformers, httpx, pdf2image |
| File Processing | Pillow, python-multipart |

---

## System Architecture

User uploads an invoice through the web interface.

The request flows through a FastAPI proxy deployed on Railway, which performs validation, deduplication, and retry handling.

The processed file is forwarded securely to a Lightning AI GPU endpoint hosting the Qwen2-VL model.

The model extracts structured invoice data and returns a normalized JSON response.

The frontend renders the extracted output in a clean tabular format.

---

## Project Structure

extracto/
├── main.py              # FastAPI proxy server and frontend integration
├── server.py            # GPU inference server using Qwen2-VL
├── Dockerfile           # Container configuration for deployment
├── requirements.txt     # Project dependencies
└── static/
    └── index.html       # Frontend interface

---

## Local Setup

### 1. Clone the repository
git clone https://github.com/Khushiiii002/Extracto.git
cd Extracto

### 2. Install dependencies
pip install -r requirements.txt

### 3. Run the FastAPI server
uvicorn main:app --reload --port 8000

### 4. Open the application
http://localhost:8000

Note: The inference server (server.py) requires a GPU environment and runs separately on Lightning AI.

---

## Author

Khushi  
GitHub: https://github.com/Khushiiii002

---

## Notes

This project is designed as a modular system with a clear separation between API handling and model inference, making it scalable and deployment-friendly across cloud GPU environments.
