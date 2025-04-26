import argparse
import base64
import cv2
import requests
import numpy as np
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "qwen/qwen2.5-vl-72b-instruct:free"

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
    img_proc = preprocess_image(image_path)
    data_uri = encode_image_to_data_uri(img_proc)

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    """You are Qwen, an expert in interpreting blackboard photographs from advanced mathematics lectures and transcribing their content into precise LaTeX code.

The image is from a lecture by Peter Scholze on Habiro cohomology, arithmetic geometry, and p-adic Hodge theory. The lecture includes a wide range of mathematical concepts, including but not limited to:

Habiro ring, number fields, Frobenius endomorphism, Bloch group, algebraic K-theory, p-adic dilogarithm, q-de Rham cohomology, étale cohomology, de Rham cohomology, crystalline cohomology, prismatic cohomology, p-adic Hodge theory, singular cohomology, pro-étale cohomology, Nygaard filtration, A Omega, adic spaces, Berkovich spaces, perfectoid spaces, formal schemes, delta rings, prisms, Kashaev invariant, Chern–Simons theory, Donaldson–Thomas invariants, cohomological Hall algebras, Kontsevich–Soibelman series, infinite Pochhammer symbol, and mathematicians such as Stavros Garoufalidis, Campbell Wheeler, Don Zagier, Bhargav Bhatt, Matthew Morrow, James Borger, André Joyal, and Alexandru Buium. You should also recognize terms like q-analogues, Legendre symbol, Tate module, Witt vectors, Milne’s sheaves, and syntomic complexes.

The blackboard may contain:

Complex formulas and equations

Definitions

Diagrams (including commutative diagrams and heuristic sketches)

Tables

Your task is to:

Accurately transcribe all visible content into LaTeX, preserving the structure and mathematical notation.

If the image contains a complex diagram (such as a cohomological tower, commutative square, or geometric visualization), describe the structure of the diagram in a LaTeX-compatible comment and include the relative image path (e.g., % diagram from board_hh_mm_ss_000.jpg). The purpose is to allow diagrams to be typeset manually later.

For any part of the board that is unclear or illegible, insert % illegible as a placeholder comment in the LaTeX output.

Output only the LaTeX code—no additional explanation, no commentary, and no natural language interpretation."""
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Here is a blackboard image. Please transcribe exactly what you see into LaTeX markup."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri}
                    }
                ]
            }
        ]
    }

    response = requests.post(ENDPOINT, headers=HEADERS, json=payload)
    if response.status_code == 200:
        reply = response.json()
        try:
            return reply["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            return "[ERROR: Unexpected LLM response format]"
    else:
        return f"[ERROR {response.status_code}]: {response.text}"

def process_boards_json(json_path):
    import json
    import os
    
    # Load the JSON file
    with open(json_path, 'r') as f:
        boards_data = json.load(f)
    
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
