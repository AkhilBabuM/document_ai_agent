import cv2
import re
import pytesseract
from pydantic import BaseModel, ValidationError, Field
from typing import List, Dict, Union
import json
from datetime import date
import asyncio
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI

from openai import OpenAI


# Initialize LLM
llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    model_name="qwen2.5-7b-instruct-1m",
    openai_api_key="sk-fake-key",
)

# LM Studio API Endpoint
LMSTUDIO_API_URL = "http://localhost:1234/v1/completions"

client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")


# Pydantic Models for Documents
class Passport(BaseModel):
    passport_number: str
    full_name: str
    nationality: str
    date_of_birth: str
    expiry_date: str

class PAN(BaseModel):
    permanent_account_number: str
    name: str
    fathers_name: str
    date_of_birth: date


class DrivingLicense(BaseModel):
    license_number: str
    full_name: str
    issue_date: date
    expiry_date: str
    category: str
    address: str = Field(..., description="Tha address, along with pincode. All lowercase")
    bloodgroup: str
    date_of_issue: date
    # pin_code: int

class Aadhaar(BaseModel):
    aadhaar_number: str
    full_name: str
    dob: date
    gender: str


# Available Document Models
document_models = {
    "passport": Passport,
    "driving_license": DrivingLicense,
    "pan": PAN,
    "aadhaar": Aadhaar
}


# State Model for LangGraph
class DocumentProcessingState(BaseModel):
    image_path: str
    extracted_text: Union[str, None] = None
    document_type: Union[str, None] = None
    extracted_data: Union[Dict, None] = None
    validated_data: Union[Dict, None] = None
    error: Union[str, None] = None


# OCR Step
async def extract_text_from_image(state: DocumentProcessingState) -> DocumentProcessingState:
    """Extract text from an image using OCR."""
    try:
        image = cv2.imread(state.image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        state.extracted_text = pytesseract.image_to_string(gray).strip()
    except Exception as e:
        state.error = f"OCR failed: {str(e)}"
    return state

async def query_lmstudio(prompt: str) -> str:
    """Query LM Studio using OpenAI-style API."""
    try:
        response = client.chat.completions.create(
            model="qwen2.5-7b-instruct-1m",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            max_tokens=1028,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error in LLM query: {str(e)}"


# Identify Document Type Step (Now with Context & One-Word Response)
async def identify_document_type(state: DocumentProcessingState) -> DocumentProcessingState:
    """Identify the document type using LLM, enforcing structured response."""
    if state.error:
        return state  # Skip if there was an error in OCR

    # Pass Pydantic Model Examples
    pydantic_examples = {
        "passport": Passport.model_json_schema(),
        "driving_license": DrivingLicense.model_json_schema(),
        "pan": PAN.model_json_schema(),
        "aadhaar": Aadhaar.model_json_schema()
    }

    prompt = f"""
    You are an AI that classifies documents based on their extracted text.
    The possible document types are: "passport", "driving_license", or "other".
    
    Here are the available document models:
    
    {json.dumps(pydantic_examples, indent=4)}

    Given this extracted text:
    
    "{state.extracted_text}"
    
    Match it with one of the available models and return only one word, which is the model name.
    If an exact match if not found, return the closest match.

    Do not return an explanation, just return a single word.
    No explanation, just single word of model name.
    """
    # prompt= "say driving_license"
    response = await query_lmstudio(prompt)
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
async def extract_relevant_data(state: DocumentProcessingState) -> DocumentProcessingState:
    """Extract structured data from the document using LLM."""
    if state.error:
        return state  # Skip processing if an error occurred

    prompt = f"""
    Extract relevant information from the following text based on the {state.document_type} model:

    "{state.extracted_text}"

    Ensure that the output follows this Pydantic model fields.

    {json.dumps(document_models[state.document_type].model_json_schema(), indent=4)}

    Return the extracted data strictly as a JSON object, such that passing it directly to the model will validate it.
    """

    response = await query_lmstudio(prompt)
    cleaned_response = clean_llm_response(response) # Remove <think> sections
    try:
        raw_data = json.loads(cleaned_response)

        # Extract only the "properties" field if it exists
        if "properties" in raw_data:
            state.extracted_data = raw_data["properties"]
        else:
            state.extracted_data = raw_data  # Use as-is if already in correct format

    except json.JSONDecodeError:
        state.error = "Failed to parse JSON from LLM response."
    return state


# Validate Data Step
async def validate_document_data(state: DocumentProcessingState) -> DocumentProcessingState:
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
async def process_document(image_path: str) -> Dict:
    """Runs the LangGraph pipeline asynchronously for a single document."""
    pipeline = build_langraph_pipeline()
    state = DocumentProcessingState(image_path=image_path)
    return await pipeline.ainvoke(state)


async def process_multiple_documents(image_paths: List[str]) -> List[Dict]:
    """Runs the LangGraph pipeline for multiple documents concurrently."""
    tasks = [process_document(image) for image in image_paths]
    return await asyncio.gather(*tasks)


if __name__ == "__main__":
    image_files = ["documents/aadhaar_1.jpeg"]

    results = asyncio.run(process_multiple_documents(image_files))

    def convert_pydantic_to_json(obj):
        """Converts Pydantic models to JSON serializable format."""
        if isinstance(obj, BaseModel):
            return obj.model_dump_json(indent=2)
        return json.dumps(obj, default=str, indent=2)

    for res in results:
        print(convert_pydantic_to_json(res))
