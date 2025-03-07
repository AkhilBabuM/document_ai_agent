# AI Agent OCR

## Overview
AI Agent OCR is an intelligent document processing pipeline that employs an **agentic approach** to Optical Character Recognition (OCR) and document validation. Leveraging **LangGraph**, this AI-driven agent orchestrates the end-to-end document processing workflow, including OCR extraction, document type detection, structured data extraction, and validation using **Large Language Models (LLMs)**. The system is designed to automatically process diverse document types, including passports, Aadhaar cards, PAN cards, and more, with minimal manual intervention.

## **Seamless Extensibility – Easily Add New Document Models**
The main feature of AI Agent OCR is its **generic and highly extensible architecture**. Defining a new document model requires minimal effort—just add a new **Pydantic model**, map it in the system, and the AI agent **automatically recognises, extracts, and validates** data for the new document type without requiring changes to the core logic.

## **Agentic Workflow**
AI Agent OCR functions as an autonomous AI agent that:
1. **Detects documents** and their types using LLM inference.
2. **Extracts structured data** from images using OCR and validation models.
3. **Validates and refines** extracted data against predefined schemas.
4. **Adapts dynamically** to new document types with minimal configuration changes.

## Adding New Document Models
The process of **adding new document types is as simple as defining a Pydantic model**. Once defined, the **LangGraph agent automatically integrates the new model into its workflow**.

### Example: Adding a New Document Model
In `models/document_models.py`:
```python
class VoterID(DateConversionMixin):
    voter_id_number: str
    full_name: str
    date_of_birth: date
    address: str
```

Now, adding the mapping:
In `models/__init__.py`:
```python
document_models = {
    "passport": Passport,
    "driving_license": DrivingLicense,
    "pan": PAN,
    "aadhaar": Aadhaar,
    "voter_id": VoterID,  # Newly added model
}
```

The system can now **automatically detect, extract, and validate voter ID documents** when provided with an image—**no additional code changes required**.

## Features
- **Agentic document processing pipeline** using LangGraph
- **OCR-based text extraction** from images (Tesseract)
- **LLM-powered document identification** (Ollama)
- **Structured data extraction and validation** using Pydantic models
- **Effortless model extensibility—just define and add**
- **FastAPI API Server** 

## Project Structure
```
ai_agent_ocr
├── agent_api.py           # FastAPI-based API for document OCR
├── agent                 # Core processing logic
│   ├── agent_pipeline.py # LangGraph-powered agentic document processing pipeline
│   ├── llm_invoke.py     # Handles interaction with LLM (Ollama)
│   ├── utils.py          
├── models                # Pydantic models for document validation
│   ├── document_models.py    # Definitions for document models (Passport, PAN, Aadhaar, etc.)
│   ├── field_validators.py   # Validators for specific fields
│   ├── processing_models.py  # Additional processing-related models
│   ├── response_models.py    # Output models for API responses
├── utils.py         
├── requirements.txt      
```

## Installation
### Prerequisites
- Python 3.10+
- Virtual environment (optional but recommended)
- Tesseract OCR installed (ensure `pytesseract` can access it)
- Ollama API running locally (`http://localhost:11434/v1`)
- FastAPI for API endpoints

### Steps
```bash
# Clone the repository
git clone https://github.com/your-repo/ai_agent_ocr.git
cd ai_agent_ocr

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn agent_api:app --host 0.0.0.0 --port 8008 --reload
```

## Usage
### API Endpoint: Document OCR
#### Request
```
POST /ocr_document
Content-Type: multipart/form-data
```
- Accepts one or more image files of supported document types

#### Response
```json
{
  "document_type": "passport",
  "validated_data": {
    "passport_number": "A1234567",
    "full_name": "John Doe",
    "nationality": "India",
    "date_of_birth": "1990-01-01",
    "expiry_date": "2030-01-01"
  },
  "error": null
}
```

## License
MIT License. See `LICENSE` for details.

## Author
**Akhil Baran**  
Email: [akhilbabu.mpn@gmail.com](mailto:akhilbabu.mpn@gmail.com)

