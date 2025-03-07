from pydantic import Field, BaseModel
from typing import Union, Any, List, Dict

class DocumentProcessingState(BaseModel):
    image_path: Union[List[str], None] = None
    extracted_text: Union[str, None] = None
    document_type: Union[str, None] = None
    extracted_data: Union[Dict, None] = None
    validated_data: Union[Dict, None] = None
    error: Union[str, None] = None