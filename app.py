import cv2
import pytesseract
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Union
import json
import os
import requests
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    model_name="deepseek-r1-distill-qwen-7b",
    openai_api_key="sk-fake-key",  # Fake key to bypass API key check
)

# LM Studio API Endpoint
LMSTUDIO_API_URL = "http://localhost:1234/v1/completions"

# Pydantic Models for Different Document Types
class Passport(BaseModel):
    passport_number: str
    full_name: str
    nationality: str
    date_of_birth: str
    expiry_date: str

class DrivingLicense(BaseModel):
    license_number: str
    full_name: str
    issue_date: str
    expiry_date: str
    category: str

# Mapping of document types to their corresponding Pydantic models
document_models = {
    "passport": Passport,
    "driving_license": DrivingLicense,
}

# OCR Function to Extract Text from Image
def extract_text_from_image(image_path: str) -> str:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

# Function to Call LM Studio LLM API
def query_lmstudio(prompt: str) -> str:
    payload = {
        "model": "lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        "prompt": prompt,
        "temperature": 0.1,
        "max_tokens": 1024
    }
    response = requests.post(LMSTUDIO_API_URL, json=payload)
    response_json = response.json()
    return response_json["choices"][0]["text"].strip()

# Function to Identify Document Type Using LLM
def identify_document_type(text: str) -> str:
    prompt = f"""
    The following text was extracted from an image:
    {text}
    Identify if this document is a passport, driving license, or another type.
    Return only the document type as a single word.
    """
    return query_lmstudio(prompt).lower()

# Function to Extract Relevant Data Based on Identified Document Type
def extract_relevant_data(text: str, document_type: str) -> Union[Dict, None]:
    prompt = f"""
    Extract the relevant fields for a {document_type} from the following text:
    {text}
    Return the extracted data as a JSON dictionary.
    """
    return json.loads(query_lmstudio(prompt))

# Function to Validate and Return Structured Data
def process_document(image_path: str) -> Union[Dict, str]:
    extracted_text = extract_text_from_image(image_path)
    document_type = identify_document_type(extracted_text)
    
    if document_type not in document_models:
        return f"Unrecognized document type: {document_type}"
    
    extracted_data = extract_relevant_data(extracted_text, document_type)
    print(f"The extracted data is: {extracted_data}")
    
    try:
        validated_data = document_models[document_type](**extracted_data)
        return validated_data.dict()
    except ValidationError as e:
        return f"Validation error: {e}"

# AI Agent Setup
def agentic_system():
    tools = [
        Tool(name="OCR", func=extract_text_from_image, description="Extract text from an image."),
        Tool(name="Identify Document Type", func=identify_document_type, description="Identify the document type."),
        Tool(name="Extract Data", func=extract_relevant_data, description="Extract structured data from a document."),
        Tool(name="Validate Data", func=process_document, description="Validate and structure document data.")
    ]
    agent = initialize_agent(tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    return agent

# Batch Processing for Multiple Images
def process_multiple_documents(image_paths: List[str]) -> List[Dict]:
    agent = agentic_system()
    results = []
    for image_path in image_paths:
        result = agent.invoke(image_path)
        results.append(result)
    return results

# Example Usage
if __name__ == "__main__":
    image_files = ["ai_pan2.jpeg"]
    results = process_multiple_documents(image_files)
    for res in results:
        print(res)
