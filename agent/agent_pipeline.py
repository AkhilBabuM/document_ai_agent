from models import DocumentProcessingState
import httpx
from PIL import Image
import pytesseract
from models import document_models
import json
from pydantic import BaseModel, ValidationError
import re
from typing import List, Dict
from langgraph.graph import StateGraph
from .llm_invoke import query_llm
from .utils import clean_llm_response, remove_newline_characters

async def extract_text_from_image(
    state: DocumentProcessingState,
) -> DocumentProcessingState:
    """Extract text from an image using OCR with preprocessing."""
    state.extracted_text = ""
    try:
        for image_path in state.image_path:
            # Load the image
            image = Image.open(image_path)

            # Extract text using OCR.
            # We will do the OCR with two configurations to maximize text detetion.
            # Non-sparse text with alignment correction
            state.extracted_text += pytesseract.image_to_string(
                image, lang="eng", config="--psm 1"
            )
            # Sparse text with alignment correction
            state.extracted_text += pytesseract.image_to_string(
                image, lang="eng", config="--psm 12"
            )
            state.extracted_text = remove_newline_characters(state.extracted_text)
            if not state.extracted_text:
                state.error = "OCR detected no text."
    except Exception as e:
        state.error = f"OCR failed: {str(e)}"

    return state

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
    No explanation, just single word of EXACT model name, among {list(document_model_schemas.keys())}
    """
    response = await query_llm(prompt)
    state.document_type = response.lower()

    if state.document_type not in document_models.keys():
        state.error = "Could not detect document type"

    return state

# Extract Data Step
async def extract_relevant_data(
    state: DocumentProcessingState,
) -> DocumentProcessingState:
    """Extract structured data from the document using LLM."""
    if state.error:
        return state  # Skip processing if an error occurred

    prompt = f"""
    I will provide the ocr of a single document, done twice with different configuration to cover all bases of text available.
    Extract relevant information from the following text based on the {state.document_type} model:

    "{state.extracted_text}"

    Ensure that the output you provide can be validated, and adheres to this Pydantic model schema.
    Make sure dates are in proper format (yyyy-mm-dd). Ignore any additional data for the dates.

    {json.dumps(document_models[state.document_type].model_json_schema(), indent=4)}

    Return the extracted data strictly as a JSON object, such that passing it directly to the model will validate it.
    """

    response = await query_llm(prompt)
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


# Invoking Document Processing Agent pipeline
async def process_document(image_path: List[str]) -> Dict:
    """Runs the LangGraph pipeline for a single document."""
    pipeline = build_langraph_pipeline()
    state = DocumentProcessingState(image_path=image_path)
    return await pipeline.ainvoke(state)