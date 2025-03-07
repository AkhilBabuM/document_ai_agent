from pydantic import BaseModel
import json

def convert_pydantic_to_json(obj):
    """Converts Pydantic models to JSON serializable format."""
    if isinstance(obj, BaseModel):
        return obj.model_dump_json(indent=2)
    return json.dumps(obj, default=str, indent=2)

