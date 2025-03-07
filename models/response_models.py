from pydantic import BaseModel
from typing import Union, Dict

class OCRResponse(BaseModel):
    document_type: Union[str, None] = None
    validated_data: Union[Dict, None] = None
    error: Union[str, None] = None