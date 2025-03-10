from typing import List

from fastapi import FastAPI, File, UploadFile
from tempfile import NamedTemporaryFile
import os
import uvicorn

from models import OCRResponse, DocumentProcessingState
from agent import process_document
from utils import convert_pydantic_to_json

app = FastAPI()


@app.post("/ocr_document")
async def ocr_document(files: List[UploadFile] = File(...)) -> OCRResponse:
    """
    Given an image or images for a single credential, 
    detects the document type and if available, validates it against the model and returns the validated ouput.
    """
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
    if not result.validated_data:
        return OCRResponse(document_type=result.document_type, error=result.error)
    return OCRResponse(
        document_type=result.document_type, validated_data=result.validated_data
    )


if __name__ == "__main__":
    uvicorn.run("agent_api:app", host="0.0.0.0", port=8008, reload=True)
