import re
from typing import List, Dict


def merge_dicts(dict_list: List[Dict]) -> Dict:
    """Merge a list of dictionaries into a single dictionary, giving precedence to the first occurrence of each key."""
    merged_dict = {}
    for d in dict_list:
        for key, value in d.items():
            if key not in merged_dict:
                merged_dict[key] = value
    return merged_dict


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
        response = match.group(1).strip()  # Return only the JSON content

    # Step 3: Extract JSON block if it exists
    match = re.search(r"```json(.*?)```", response, re.DOTALL)
    if match:
        response = match.group(1).strip()  # Return only the JSON content

    cleaned_response = remove_newline_characters(response.strip())  # Return the cleaned text
    return cleaned_response

def remove_newline_characters(text: str) -> str:
    """Removes all '\n' and '\\n' and replaces it with ' ' blanks."""
    cleaned_text = text.replace("\n", " ").replace("\\n", " ")
    return cleaned_text