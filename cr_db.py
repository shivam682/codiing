# extract_cr_info.py
import os
import json
from typing import Dict
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# LLM initialization (can change temperature or model_name as needed)
llm = ChatOpenAI(temperature=0, model_name="gpt-4")

EXTRACTION_PROMPT = """
You will be given a CR document as plain text. Extract the following fields:
- logs_checked: List of log file names or log types that were analyzed.
- issue_summary: A concise one-line summary of what the root issue was.
- build_version: Firmware or build version if mentioned.
- resolution_summary: How the issue was resolved (e.g., code fix, config change).

Return the answer as a JSON object with keys: logs_checked, issue_summary, build_version, resolution_summary.
"""

def extract_cr_structured_info(txt_path: str) -> Dict:
    """Extract relevant metadata from raw CR txt using LLM"""
    with open(txt_path, 'r', errors='ignore') as file:
        content = file.read()

    message = HumanMessage(content=f"{EXTRACTION_PROMPT}\n\n{content}")
    response = llm([message])

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON for {txt_path}. Raw response:\n{response.content}")
        result = {}

    return result

def process_cr_folder(folder_path: str, output_path: str):
    """Process all CR txt files in a folder and write structured JSON metadata"""
    all_cr_data = {}
    for fname in os.listdir(folder_path):
        if fname.endswith(".txt"):
            cr_id = os.path.splitext(fname)[0]
            full_path = os.path.join(folder_path, fname)
            metadata = extract_cr_structured_info(full_path)
            all_cr_data[cr_id] = metadata

    with open(output_path, 'w') as outfile:
        json.dump(all_cr_data, outfile, indent=2)

# Example usage:
# process_cr_folder("/path/to/cr_texts", "parsed_cr_info.json")
