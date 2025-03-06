import re
import pytesseract
from pydantic import BaseModel, ValidationError, Field, field_validator
from typing import List, Dict, Union, Any
import json
from datetime import date, datetime
import asyncio
from langgraph.graph import StateGraph
from PIL import Image
import httpx

from openai import OpenAI

# Ollama API Endpoint
OLLAMA_API_URL = "http://localhost:11434/v1"

client = OpenAI(base_url=OLLAMA_API_URL, api_key="test")


# Pydantic Models for Documents
date_formats = (
    "%Y-%m-%d",  # 2024-06-25 (ISO 8601 Standard)
    "%d/%m/%Y",  # 25/06/2024 (Common in India, UK)
    "%m/%d/%Y",  # 06/25/2024 (US Format)
    "%d-%m-%Y",  # 25-06-2024 (Common in India, UK)
    "%Y/%m/%d",  # 2024/06/25 (Rare but possible)
    "%m-%d-%Y",  # 06-25-2024 (US Format)
    "%d %b %Y",  # 25 Jun 2024 (Short Month Name)
    "%d %B %Y",  # 25 June 2024 (Full Month Name)
    "%b %d, %Y",  # Jun 25, 2024 (US, Passport Style)
    "%B %d, %Y",  # June 25, 2024 (Long-form US, UK)
    "%d.%m.%Y",  # 25.06.2024 (German, European style)
    "%m.%d.%Y",  # 06.25.2024 (Alternative US style)
    "%Y.%m.%d",  # 2024.06.25 (Database formats)
    "%d%m%Y",  # 25062024 (No separator, found in OCR errors)
    "%Y%m%d",  # 20240625 (Machine-readable)
    "%d-%b-%Y",  # 25-Jun-2024 (Found in some Indian credentials)
    "%d-%B-%Y",  # 25-June-2024 (Long form, uncommon)
    "%Y %b %d",  # 2024 Jun 25 (Seen in some passport formats)
    "%Y %B %d",  # 2024 June 25
    "%b-%d-%Y",  # Jun-25-2024
    "%B-%d-%Y",  # June-25-2024
    "%b %d %Y",  # Jun 25 2024
    "%B %d %Y",  # June 25 2024
    "%d/%b/%Y",  # 25/Jun/2024 (Common in travel documents)
    "%d/%B/%Y",  # 25/June/2024
    "%m-%Y-%d",  # 06-2024-25
)


class DateConversionMixin(BaseModel):
    """Mixin that attempts to convert values to dates, but passes them through if conversion fails."""

    @field_validator("*", mode="before", check_fields=False)
    @classmethod
    def try_convert_to_date(cls, value: Any) -> Any:
        """Attempt to convert the value to a date. If unsuccessful, return the original value."""

        # If already a date, return as is
        if isinstance(value, date):
            return value

        # Try parsing valid date strings
        if isinstance(value, str):
            for fmt in date_formats:
                try:
                    return str(datetime.strptime(value, fmt).date())  # Convert to date
                except ValueError:
                    continue  # Try next format
            return value  # If no formats match, return the original string

        # Handle integer timestamps (convert to date)
        if isinstance(value, int):
            try:
                return str(datetime.fromtimestamp(value).date())
            except ValueError:
                return value  # Return as is if not a valid timestamp

        return value  # Return original value if it can't be converted


# Pydantic Models for Documents
class Passport(DateConversionMixin):
    passport_number: str
    full_name: str
    nationality: str
    date_of_birth: date
    expiry_date: date


class PAN(DateConversionMixin):
    permanent_account_number: str
    name: str
    fathers_name: str
    date_of_birth: date


class DrivingLicense(DateConversionMixin):
    license_number: Union[str, None] = None
    full_name: Union[str, None] = None
    issue_date: Union[date, None] = None
    expiry_date: Union[date, None] = None
    category: Union[List[str], None] = Field(
        None, description="can have multiple classes"
    )
    address: Union[str, None] = None
    bloodgroup: Union[str, None] = Field(None, description="must be valid blood type")
    son_of: Union[str, None] = Field(None, description="given as s/o in credentials")
    date_of_issue: Union[date, None] = None


class Aadhaar(DateConversionMixin):
    aadhaar_number: str
    full_name: str
    dob: date
    gender: str


# Available Document Models
document_models = {
    "passport": Passport,
    "driving_license": DrivingLicense,
    "pan": PAN,
    "aadhaar": Aadhaar,
}


# State Model for LangGraph
class DocumentProcessingState(BaseModel):
    image_path: Union[List[str], None] = None
    extracted_text: Union[str, None] = None
    document_type: Union[str, None] = None
    extracted_data: Union[Dict, None] = None
    validated_data: Union[Dict, None] = None
    error: Union[str, None] = None


class OCRResponse(BaseModel):
    validated_data: Union[Dict, None] = None


async def extract_text_from_image(
    state: DocumentProcessingState,
) -> DocumentProcessingState:
    """Extract text from an image using OCR with preprocessing."""
    state.extracted_text = ""
    try:
        for image_path in state.image_path:
            # Load the image
            image = Image.open(image_path)

            # Extract text using OCR
            state.extracted_text += pytesseract.image_to_string(
                image, lang="eng", config="--psm 11"
            )
            if not state.extracted_text:
                state.error = "OCR detected no text."
    except Exception as e:
        state.error = f"OCR failed: {str(e)}"

    return state


async def query_ollama(prompt: str) -> str:
    """Query Ollama's locally running model."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/generate",  # Ollama's API
                json={"model": "qwen2.5", "prompt": prompt, "stream": False},
                timeout=60,
            )
            response_data = response.json()

            if "response" in response_data:
                return response_data["response"].strip()
            else:
                return f"Error: Unexpected response format {response_data}"
    except Exception as e:
        return f"Error in LLM query: {str(e)}"


def merge_dicts(dict_list: List[Dict]) -> Dict:
    """Merge a list of dictionaries into a single dictionary, giving precedence to the first occurrence of each key."""
    merged_dict = {}
    for d in dict_list:
        for key, value in d.items():
            if key not in merged_dict:
                merged_dict[key] = value
    return merged_dict


def convert_pydantic_to_json(obj):
    """Converts Pydantic models to JSON serializable format."""
    if isinstance(obj, BaseModel):
        return obj.model_dump_json(indent=2)
    return json.dumps(obj, default=str, indent=2)


# Identify Document Type Step (Now with Context & One-Word Response)
async def identify_document_type(
    state: DocumentProcessingState,
) -> DocumentProcessingState:
    """Identify the document type using LLM, enforcing structured response."""
    if state.error:
        return state  # Skip if there was an error in OCR

    # Pydantic Model Scehmas for available documents
    document_model_schemas = {
        model_name: model.model_json_schema()
        for model_name, model in document_models.items()
    }

    prompt = f"""
    You are an AI that classifies documents based on their extracted text.
    The possible document types are provided below.
    
    Here are the available document models:
    
    {json.dumps(document_model_schemas, indent=4)}

    Given this extracted text:
    
    "{state.extracted_text}"
    
    Match it with one of the available models and return only one word, which is the model name.
    If an exact match is not found, return the closest match.

    Do not return an explanation, just return a single word.
    No explanation, just single word of model name.
    """
    response = await query_ollama(prompt)
    state.document_type = response

    return state


def clean_llm_response(response: str) -> str:
    """
    Extracts JSON content from LLM response.

    - Removes any <think>...</think> sections.
    - Extracts JSON content from Markdown code blocks (```json ... ```).

    :param response: The raw LLM response as a string.
    :return: The cleaned JSON string.
    """
    if not response:
        return ""

    # Step 1: Remove <think> sections if present
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    # Step 2: Extract JSON block if it exists
    match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
    if match:
        return match.group(1).strip()  # Return only the JSON content

    return response.strip()  # Return the cleaned text


# Extract Data Step
async def extract_relevant_data(
    state: DocumentProcessingState,
) -> DocumentProcessingState:
    """Extract structured data from the document using LLM."""
    if state.error:
        return state  # Skip processing if an error occurred

    prompt = f"""
    Extract relevant information from the following text based on the {state.document_type} model:

    "{state.extracted_text}"

    Ensure that the output you provide can be validated, and adheres to this Pydantic model schema.
    Make sure dates are in proper format.

    {json.dumps(document_models[state.document_type].model_json_schema(), indent=4)}

    Return the extracted data strictly as a JSON object, such that passing it directly to the model will validate it.
    """

    response = await query_ollama(prompt)
    cleaned_response = clean_llm_response(response)  # Remove <think> sections
    try:
        raw_data = json.loads(cleaned_response)

        # If multiple documents are detected, consider the first one.
        if isinstance(raw_data, list):
            raw_data = raw_data[0]

        # Extract only the "properties" field if it exists
        if "properties" in raw_data:
            state.extracted_data = raw_data["properties"]
        else:
            state.extracted_data = raw_data  # Use as-is if already in correct format

    except json.JSONDecodeError:
        state.error = "Failed to parse JSON from LLM response."
    return state


# Validate Data Step
async def validate_document_data(
    state: DocumentProcessingState,
) -> DocumentProcessingState:
    """Validate and structure extracted data using Pydantic models."""
    if state.error:
        return state  # Skip processing if an error occurred

    if state.document_type not in document_models:
        state.error = f"Unrecognized document type: {state.document_type}"
        return state

    try:
        validated_data = document_models[state.document_type](**state.extracted_data)
        state.validated_data = validated_data.model_dump()
    except ValidationError as e:
        state.error = f"Validation error: {e}"

    return state


# LangGraph Workflow
def build_langraph_pipeline():
    """Builds the LangGraph workflow for document processing."""
    graph = StateGraph(DocumentProcessingState)

    graph.add_node("OCR", extract_text_from_image)
    graph.add_node("Identify Document Type", identify_document_type)
    graph.add_node("Extract Data", extract_relevant_data)
    graph.add_node("Validate Data", validate_document_data)

    graph.add_edge("OCR", "Identify Document Type")
    graph.add_edge("Identify Document Type", "Extract Data")
    graph.add_edge("Extract Data", "Validate Data")

    graph.set_entry_point("OCR")
    graph.set_finish_point("Validate Data")

    return graph.compile()


# Asynchronous Document Processing
async def process_document(image_path: List[str]) -> Dict:
    """Runs the LangGraph pipeline asynchronously for a single document."""
    pipeline = build_langraph_pipeline()
    state = DocumentProcessingState(image_path=image_path)
    return await pipeline.ainvoke(state)

from fastapi import FastAPI, File, UploadFile
from tempfile import NamedTemporaryFile
import os
import uvicorn

app = FastAPI()


@app.post("/ocr_document")
async def ocr_document(files: List[UploadFile] = File(...)) -> OCRResponse:
    image_paths = []
    try:
        for file in files:
            contents = await file.read()
            file_extension = os.path.splitext(file.filename)[1]
            with NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                tmp.write(contents)
                image_paths.append(tmp.name)

        result = await process_document(image_path=image_paths)
        result = DocumentProcessingState.model_validate(result)
    finally:
        for path in image_paths:
            try:
                os.remove(path)
            except OSError as e:
                print(f"Error deleting temporary file {path}: {e}")

    print(convert_pydantic_to_json(result))
    return OCRResponse(validated_data=result.validated_data)


if __name__ == "__main__":
    uvicorn.run("agent_api:app", host="0.0.0.0", port=8008, reload=True)
