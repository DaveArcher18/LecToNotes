import argparse
import base64
import cv2
import requests
import numpy as np
import os
import time
import re
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
OCR_MODEL = "qwen/qwen2.5-vl-72b-instruct:free"
REFINER_MODEL = "qwen/qwen3-14b:free"
THROTTLETIME = 0 


def call_openrouter_api(messages, model):
    """Generic function to call OpenRouter APIs."""
    api_key = os.getenv("OPENROUTER_API_KEY")  # Secure your API key
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    try:
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except (ValueError, requests.exceptions.JSONDecodeError):
        # Fallback: return raw text if JSON parsing fails
        return response.text.strip()

def postprocess_text(text):
    """Final minor corrections: fix unicode, clean LaTeX fencing, normalize spacing."""
    try:
        # use the hyphen form so Python doesn’t interpret any “\u” or “\e” in the literal
        text = text.encode('utf-8').decode('unicode-escape')
        # Fix Latin-1 misencoded sequences from UTF-8 escapes (e.g., Ã© → é)
        text = text.encode('latin-1').decode('utf-8')
    except UnicodeDecodeError:
        # Leave original text if decoding fails
        pass
    text = re.sub(r'text{([^}]*)}', r'\\text{\1}', text)
    text = re.sub(r'(?<!\$)\s*\\begin{array}', r'$$\n\\begin{array}', text)
    text = re.sub(r'\\end{array}(?!\$)', r'\\end{array}\n$$', text)
    text = text.replace('```markdown', '').replace('```', '')
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def refine_text_via_llm(text):
    """Send text to a second-pass LLM refiner."""
    prompt = rf"""
You are an expert Markdown formatter specializing in mathematics.

Your output will be displayed directly on a user-facing website, so it must be polished, production-quality, and visually clear.

Given the following content combining Markdown and LaTeX, your tasks are:

- Preserve all mathematical symbols and structures exactly.
- Correct any Markdown syntax errors and ensure valid Markdown.
- Decode any Unicode escape sequences within text environments (for example, convert `\u00c3\u00a9` to `é`) before final output.
- After decoding, fix any Latin-1 misencoded sequences (for example, convert `Ã©` to `é`).
- Wrap LaTeX math environments (like \begin{{array}}...\end{{array}}) inside $$...$$ fences.
- Use plain Markdown lists, tables, and headings wherever possible.
- Do not change mathematical meaning.
- Do not add commentary or explanation.
- Output only clean production-quality Markdown.
- If the content to refine is exactly '%illegible', output it unchanged.

Content to refine:

{text}
"""
    messages = [
        {"role": "system", "content": "You are a precise Markdown and LaTeX formatter."},
        {"role": "user", "content": prompt}
    ]
    return call_openrouter_api(messages, REFINER_MODEL)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (1024, 1024))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (very light enhancement)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Convert back to 3 channel image
    final = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    # Show the preprocessed image for inspection
    # plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    # plt.title("Preprocessed Image")
    # plt.axis("off")
    # plt.show()

    return final

def encode_image_to_data_uri(img):
    success, buffer = cv2.imencode(".jpg", img)
    if not success:
        raise ValueError("Failed to encode image to JPEG")
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def call_llm_with_image(image_path):
    """Processes a single board image: OCR -> Refinement -> Postprocessing."""

    # Step 0: Preprocess the image
    img_proc = preprocess_image(image_path)
    data_uri = encode_image_to_data_uri(img_proc)

    # Step 1: OCR Model
    ocr_payload = {
        "model": OCR_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
"""You are Qwen, an expert in interpreting blackboard photographs from advanced mathematics lectures and summarizing their content into **clear, readable Markdown**, using mathematical symbols and notation wherever possible.

The image is from a lecture by Peter Scholze on Habiro cohomology, arithmetic geometry, and p-adic Hodge theory.

Your tasks:
1. First, assess the complexity of the blackboard image and estimate how difficult it would be to transcribe it faithfully into Markdown.
2. If it is tractable (containing primarily text, equations, or simple diagrams), transcribe the content accurately into Markdown, using LaTeX notation where appropriate.
3. If it is highly complex (containing dense diagrams, detailed drawings, or large tables), provide a clear, concise summary highlighting the main structures, elements, and mathematical ideas rather than a full transcription.
4. If the board is blank or unreadable, output %illegible.
Output only the resulting Markdown or summary; do not include any commentary on your own process."""
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here is a blackboard image. Please transcribe exactly what you see into Markdown."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri
                        }
                    }
                ]
            }
        ]
    }

    ocr_response = call_openrouter_api(ocr_payload["messages"], OCR_MODEL)
    time.sleep(THROTTLETIME)

    # Step 2: Refinement Model
    refined_text = refine_text_via_llm(ocr_response)
    time.sleep(THROTTLETIME)

    # Step 3: Python Postprocessing
    final_text = postprocess_text(refined_text)

    return final_text
    """Processes a single board image: OCR -> Refinement -> Postprocessing."""
    with open(image_path, "rb") as image_file:
        image_data = preprocess_image(image_path)
        image_data = encode_image_to_data_uri(image_data)


    # Step 1: OCR Model
    ocr_payload = {
        "model": OCR_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
"""You are Qwen, an expert in interpreting blackboard photographs from advanced mathematics lectures and summarizing their content into **clear, readable Markdown**, using mathematical symbols and notation wherever possible.

The image is from a lecture by Peter Scholze on Habiro cohomology, arithmetic geometry, and p-adic Hodge theory.

Your tasks:
1. Accurately transcribe visible content into Markdown.
2. If the blackboard is too complicated to transcribe easily, summarize it clearly.
3. If the board is blank, output %illegible.
Output only Markdown, no commentary."""
                )
            },
            {"role": "user", "content": image_data}
        ]
    }

    ocr_response = call_openrouter_api(ocr_payload["messages"], OCR_MODEL)


    # Step 2: Refinement Model
    refined_text = refine_text_via_llm(ocr_response)
    time.sleep(THROTTLETIME)

    # Step 3: Python Postprocessing
    final_text = postprocess_text(refined_text)

    return final_text

def process_boards_json(json_path):
    import json
    import os
    
    # Load the JSON file, with fallback to quote unquoted keys
    with open(json_path, 'r') as f:
        content = f.read()
    try:
        boards_data = json.loads(content)
    except json.JSONDecodeError:
        # Quote unquoted property names, strip comments, and remove trailing commas, then retry
        fixed = re.sub(r'(?m)^[ \t]*([A-Za-z_][A-Za-z0-9_]*)[ \t]*:', r'"\1":', content)
        # Remove single-line comments
        fixed = re.sub(r'(?m)//.*$', '', fixed)
        fixed = re.sub(r'(?m)#.*$', '', fixed)
        # Remove trailing commas before } or ]
        fixed = re.sub(r',\s*(?=[}\]])', '', fixed)
        boards_data = json.loads(fixed)
    
    # Get the base directory of the JSON file
    json_dir = os.path.dirname(json_path)
    
    # Count entries that need processing
    entries_to_process = [board for board in boards_data if 'text' not in board]
    total_entries = len(boards_data)
    skipped_entries = total_entries - len(entries_to_process)
    
    if skipped_entries > 0:
        print(f"Found {skipped_entries} already processed entries with text. Will skip these.")
    
    # Process each board entry that doesn't already have text
    processed_count = 0
    for i, board in enumerate(boards_data):
        # Skip if this entry already has text
        if 'text' in board:
            continue
            
        # Get the image path - handle the path correctly
        # The path in boards.json could be relative (e.g., 'boards/filename.jpg')
        # We need to resolve it relative to the JSON file's directory
        image_path = board['path']
        
        # If the path is not absolute, make it relative to the JSON directory
        if not os.path.isabs(image_path):
            # Handle the case where the path includes 'boards/' prefix and we're already in the boards directory
            if os.path.basename(json_dir) == 'boards' and image_path.startswith('boards/'):
                image_path = image_path[len('boards/'):]  # Remove the 'boards/' prefix
            
            # Join with the JSON directory to get the absolute path
            image_path = os.path.join(json_dir, image_path)
        
        # Skip if the image doesn't exist
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} does not exist. Skipping.")
            continue
        
        # Process the image and get the LaTeX text
        processed_count += 1
        print(f"Processing image {processed_count}/{len(entries_to_process)} (entry {i+1}/{total_entries}): {os.path.basename(image_path)}")
        latex_text = call_llm_with_image(image_path)
        
        # Add the text field to the board entry
        board['text'] = latex_text
        
        # Save the updated JSON file after each successful API call
        # This ensures we don't lose progress if an error occurs later
        with open(json_path, 'w') as f:
            json.dump(boards_data, f, indent=2)
        print(f"Saved progress to {json_path} after processing image {processed_count}/{len(entries_to_process)}")
    
    if processed_count == 0:
        print(f"\nNo new entries to process. All {total_entries} entries already have text.")
    else:
        print(f"\nProcessing complete. Processed {processed_count} new entries in {json_path}.")
        print(f"Total entries: {total_entries}, Previously processed: {skipped_entries}, Newly processed: {processed_count}")

def main():
    parser = argparse.ArgumentParser(
        description="Process blackboard images and transcribe content to LaTeX."
    )
    parser.add_argument("input_path", 
                      help="Path to either a single image or a boards.json file")
    args = parser.parse_args()
    
    # Check if the input is a JSON file or an image
    if args.input_path.endswith('.json'):
        process_boards_json(args.input_path)
    else:
        # Original functionality for single image
        latex_output = call_llm_with_image(args.input_path)
        print("\n--- LaTeX Output ---\n")
        print(latex_output)

if __name__ == "__main__":
    main()
