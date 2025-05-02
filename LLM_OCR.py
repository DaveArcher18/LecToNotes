import argparse
import base64
import cv2
import requests
import numpy as np
import os
import time
import re
import json
import pickle
from pathlib import Path
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
# Update to Llama 4 Maverick for OCR
OCR_MODEL = "meta-llama/llama-4-maverick:free"
# Keep using Qwen3 for refinement (good at LaTeX cleanup)
REFINER_MODEL = "qwen/qwen3-14b:free"
# Final validator for LaTeX syntax
VALIDATOR_MODEL = "meta-llama/llama-4-maverick:free"
THROTTLETIME = 0 

# Common mathematical structure patterns to validate
MATH_PATTERNS = {
    "matrix": r"\\begin{(pmatrix|bmatrix|vmatrix|matrix)}(.*?)\\end{\1}",
    "align": r"\\begin{align}(.*?)\\end{align}",
    "equation": r"\\begin{equation}(.*?)\\end{equation}",
    "cases": r"\\begin{cases}(.*?)\\end{cases}",
    "array": r"\\begin{array}(.*?)\\end{array}",
}

def call_openrouter_api(messages, model, temperature=0.0, max_tokens=None, response_format=None):
    """Enhanced function to call OpenRouter APIs with more parameters."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }
    
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
        
    if response_format is not None:
        payload["response_format"] = response_format
    
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    try:
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except (ValueError, requests.exceptions.JSONDecodeError):
        # Fallback: return raw text if JSON parsing fails
        return response.text.strip()

def optimize_image_for_ocr(image):
    """
    Optimize an image specifically for OCR processing.
    This function can be applied to either a file path or a numpy image array.
    """
    # Handle both file path and numpy array inputs
    if isinstance(image, str):
        # It's a file path
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Could not read image from {image}")
    else:
        # It's already a numpy array
        img = image.copy()
    
    # Resize with aspect ratio preservation
    max_dim = 1024
    h, w = img.shape[:2]
    if h > w:
        new_h, new_w = max_dim, int(max_dim * w / h)
    else:
        new_h, new_w = int(max_dim * h / w), max_dim
    img = cv2.resize(img, (new_w, new_h))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to better handle chalk/marker on blackboard
    # This helps with contrast differences across the board
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, 15
    )
    
    # Detect text regions using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Convert back to normal polarity (white text on black background)
    morph = cv2.bitwise_not(morph)
    
    # Enhance contrast for better readability
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Create a color image for the model by combining original with enhanced areas
    # This gives us color context while maintaining enhanced text visibility
    final = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    # Additional noise reduction
    final = cv2.fastNlMeansDenoisingColored(final, None, 10, 10, 7, 21)
    
    return final

def encode_image_to_data_uri(img):
    """Convert OpenCV image to base64 data URI."""
    success, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise ValueError("Failed to encode image to JPEG")
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def extract_latex_with_ocr(image_data_uri):
    """First pass: Extract raw content from image using Llama 4 Maverick."""
    ocr_system_prompt = """You are an expert mathematics OCR system specializing in extracting LaTeX from blackboard images of advanced mathematics lectures.

CONTEXT: These images come from lectures on topics like Habiro cohomology, arithmetic geometry, p-adic Hodge theory, and other advanced mathematics.

YOUR TASK: Convert the blackboard content into precise, properly formatted LaTeX/Markdown, following these guidelines:

1. Mathematical Notation:
   - Use proper LaTeX for ALL mathematical expressions
   - Correctly capture superscripts, subscripts, fractions, and special symbols
   - Preserve the exact mathematical meaning and structure
   - Handle matrices, commutative diagrams, and complex equations correctly
   - Always use {2} for squared terms, NOT \\2
   - Always use _{2} for subscript 2, NOT _\\2 or _2

2. Structure:
   - Maintain the spatial layout and organization of content when meaningful
   - Use proper LaTeX environments (align, matrix, cases, etc.) for structured elements
   - For multi-line equations, use aligned environments rather than separate equations
   - For section titles, use Markdown headings (## for sections, ### for subsections)
   - For inline math, use $...$ notation
   - For display math, use $$...$$ notation
   - Use \\text{} for text within math environments

3. Special Cases:
   - For handwritten diagrams too complex to represent in LaTeX, describe them concisely
   - For unreadable or ambiguous portions, use [?] to indicate uncertainty
   - If the board is blank or completely unreadable, output %illegible

4. Format Guidelines:
   - Always surround math environments with proper delimiters
   - Always close all environments that you open
   - Use \\mathbb{R}, \\mathbb{Z}, etc. for number sets
   - Use \\text{} for text within math expressions
   - Use proper spacing in LaTeX expressions

IMPORTANT: Focus on accurately transcribing the mathematical content rather than interpreting or explaining it. Preserve all mathematical symbols exactly as they appear.
"""

    ocr_messages = [
        {"role": "system", "content": ocr_system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": "Transcribe the mathematical content from this blackboard image into precise LaTeX/Markdown."},
            {"type": "image_url", "image_url": {"url": image_data_uri}}
        ]}
    ]
    
    return call_openrouter_api(ocr_messages, OCR_MODEL, temperature=0.1, max_tokens=1500)

def structure_latex_content(raw_text):
    """Second pass: Structure the raw content into proper LaTeX environments."""
    structure_prompt = f"""You are an expert LaTeX formatter specializing in mathematical notation. Your task is to take the raw mathematical content extracted from a blackboard and structure it into proper LaTeX environments.

Raw extracted content:
```
{raw_text}
```

Please transform this into well-structured LaTeX by:

1. Identifying mathematical structures (equations, matrices, etc.) and ensuring they use the correct LaTeX environments
2. Ensuring all mathematical expressions are properly delimited with $ or $$
3. Using appropriate environments like align, matrix, cases, etc.
4. Ensuring all environments are properly opened and closed
5. Ensuring proper nesting of delimiters and environments
6. Fixing common OCR errors in mathematical notation

Specifically:
- Convert all instances of \\\\2 to {{2}} for squared terms
- Convert _2 to _{{2}} for subscripts
- Fix any unescaped special characters like % _ & # etc.
- Replace \\\\section{{}} and \\\\subsection{{}} with markdown ## and ### respectively
- Ensure all \\\\begin{{environment}} have matching \\\\end{{environment}}
- Fix any mismatched $$...$ or $...$$

OUTPUT ONLY the structured LaTeX content with no explanation or commentary.
"""

    structure_messages = [
        {"role": "system", "content": "You are an expert LaTeX formatter."},
        {"role": "user", "content": structure_prompt}
    ]
    
    return call_openrouter_api(structure_messages, REFINER_MODEL, temperature=0.1)

def validate_latex_syntax(structured_text):
    """Third pass: Validate and correct LaTeX syntax."""
    validation_prompt = f"""You are an expert LaTeX validator. Your task is to check the following mathematical content for LaTeX syntax errors and correct them.

Content to validate:
```
{structured_text}
```

Please focus on these specific issues:
1. Check that all LaTeX environments are properly opened and closed
2. Ensure all mathematical delimiters ($, $$, \\begin, \\end) are correctly paired
3. Verify that all commands have their required arguments
4. Fix any common errors with subscripts (_) and superscripts (^)
5. Convert all instances of \\2, \\3, etc. to {{2}}, {{3}}, etc.
6. Fix common spacing issues in LaTeX expressions
7. Fix Unicode or special character encoding issues
8. Replace \\section{{...}} with ## ... and \\subsection{{...}} with ### ...
9. Ensure inline math formulas use single $ and display formulas use $$
10. Fix any LaTeX commands that are incorrectly escaped (e.g., \\\\mathbb should be \\mathbb)

Common patterns to fix:
- Replace _2 with _{{2}}
- Replace ^2 with ^{{2}}
- Replace \\2 with {{2}}
- Replace \\\\begin with \\begin
- Replace \\\\end with \\end
- Fix any improperly closed environments

Return ONLY the corrected content without any explanations.
"""

    validation_messages = [
        {"role": "system", "content": "You are an expert LaTeX syntax validator."},
        {"role": "user", "content": validation_prompt}
    ]
    
    return call_openrouter_api(validation_messages, VALIDATOR_MODEL, temperature=0.0)

def enhanced_postprocess_text(text):
    """Enhanced postprocessing with specific rules for mathematical notation."""
    try:
        # Decode unicode escapes
        text = text.encode('utf-8').decode('unicode-escape')
        # Fix Latin-1 misencoded sequences from UTF-8 escapes
        text = text.encode('latin-1').decode('utf-8', errors='replace')
    except UnicodeDecodeError:
        # Leave original text if decoding fails
        pass
    
    # Clean up code blocks and markdown formatting
    text = text.replace('```latex', '').replace('```markdown', '').replace('```', '')
    
    # Fix common LaTeX issues
    # Correct \text usage
    text = re.sub(r'\\text\s+{', r'\\text{', text)
    text = re.sub(r'text{([^}]*)}', r'\\text{\1}', text)
    
    # Ensure proper math delimiters
    text = re.sub(r'(?<!\$)\\begin{array}', r'$$\n\\begin{array}', text)
    text = re.sub(r'\\end{array}(?!\$)', r'\\end{array}\n$$', text)
    text = re.sub(r'(?<!\$)\\begin{pmatrix}', r'$$\n\\begin{pmatrix}', text)
    text = re.sub(r'\\end{pmatrix}(?!\$)', r'\\end{pmatrix}\n$$', text)
    text = re.sub(r'(?<!\$)\\begin{bmatrix}', r'$$\n\\begin{bmatrix}', text)
    text = re.sub(r'\\end{bmatrix}(?!\$)', r'\\end{bmatrix}\n$$', text)
    text = re.sub(r'(?<!\$)\\begin{vmatrix}', r'$$\n\\begin{vmatrix}', text)
    text = re.sub(r'\\end{vmatrix}(?!\$)', r'\\end{vmatrix}\n$$', text)
    text = re.sub(r'(?<!\$)\\begin{cases}', r'$$\n\\begin{cases}', text)
    text = re.sub(r'\\end{cases}(?!\$)', r'\\end{cases}\n$$', text)
    text = re.sub(r'(?<!\$)\\begin{align}', r'$$\n\\begin{align}', text)
    text = re.sub(r'\\end{align}(?!\$)', r'\\end{align}\n$$', text)
    
    # Fix subscript and superscript notation - a common OCR error is \2 instead of {2}
    text = re.sub(r'([^\\])_([0-9])', r'\1_{\2}', text)  # Fix simple numeric subscripts without braces
    text = re.sub(r'([^\\])\^([0-9])', r'\1^{\2}', text)  # Fix simple numeric superscripts without braces
    text = re.sub(r'\\([0-9]+)', r'{\1}', text)  # Fix \2 to {2}
    
    # Fix sections and subsections
    text = re.sub(r'\\section{', r'## ', text)
    text = re.sub(r'\\subsection{', r'### ', text)
    text = re.sub(r'\\section\*{', r'## ', text)
    text = re.sub(r'\\subsection\*{', r'### ', text)
    text = re.sub(r'}\s*$', '', text, flags=re.MULTILINE)  # Remove closing braces at end of lines
    
    # Fix accidental escaping of LaTeX commands
    text = re.sub(r'\\\\([a-zA-Z]+)', r'\\\1', text)
    
    # Fix common OCR errors in mathematical notation
    text = text.replace('\\u0007', '')  # Remove control characters
    text = text.replace('\\u0007pprox', '\\approx')  # Fix common error for approximation symbol
    text = text.replace('\\u0007cute{e}t', '\\acute{e}t')  # Fix common error for acute accent
    
    # Fix spacing issues
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Ensure proper display math vs. inline math
    # Convert standalone equations to display math if not already
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if re.match(r'^\s*\\[a-zA-Z]+', line) and not re.search(r'\$\$|\$', line):
            lines[i] = f"$${line}$$"
    text = '\n'.join(lines)
    
    # Ensure consistent spacing around math delimiters
    text = re.sub(r'([^\s])\$\$', r'\1 $$', text)
    text = re.sub(r'\$\$([^\s])', r'$$ \1', text)
    
    # Fix mismatched or missing closing delimiters
    open_delims = text.count('$$')
    if open_delims % 2 != 0:
        text += '\n$$'  # Add closing delimiter if missing
    
    return text.strip()

def get_ocr_optimized_image(image_path_or_array):
    """Get an OCR-optimized version of the image."""
    # Optimize the image for OCR
    ocr_img = optimize_image_for_ocr(image_path_or_array)
    return ocr_img

def multi_stage_ocr_process(image_path_or_array):
    """
    Implement the multi-stage OCR pipeline.
    Can accept either a file path or a pre-loaded image array.
    """
    # Step 1: Get OCR-optimized image
    img_proc = get_ocr_optimized_image(image_path_or_array)
    data_uri = encode_image_to_data_uri(img_proc)
    
    # Step 2: First pass OCR with Llama 4 Maverick
    print("Extracting raw content...")
    raw_text = extract_latex_with_ocr(data_uri)
    
    # Skip further processing if illegible
    if raw_text.strip() == "%illegible":
        return "%illegible"
    
    # Step 3: Structure the content
    print("Structuring content...")
    structured_text = structure_latex_content(raw_text)
    
    # Step 4: Validate LaTeX syntax
    print("Validating LaTeX syntax...")
    validated_text = validate_latex_syntax(structured_text)
    
    # Step 5: Enhanced postprocessing
    print("Final postprocessing...")
    final_text = enhanced_postprocess_text(validated_text)
    
    return final_text

def check_for_ocr_data(json_path):
    """Check if there's OCR-optimized data available for the boards."""
    # Look for the pickle file with OCR-optimized images
    json_dir = os.path.dirname(json_path)
    pickle_path = os.path.join(json_dir, 'boards_ocr_data.pkl')
    
    if os.path.exists(pickle_path):
        print(f"Found OCR-optimized data at {pickle_path}")
        try:
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading OCR data: {e}")
    
    return None

def process_boards_json(json_path):
    """Process all boards in a JSON file."""
    # Check if we have OCR-optimized data
    ocr_data = check_for_ocr_data(json_path)
    
    # If we have OCR data, we'll use the pre-optimized images
    # Otherwise we'll load from the board images and optimize them ourselves
    using_preoptimized = ocr_data is not None
    
    # Load the JSON file
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
        
        # Get the corresponding entry from OCR data if available
        ocr_img = None
        if using_preoptimized:
            # Find the matching entry in ocr_data
            for ocr_entry in ocr_data:
                if ocr_entry['timestamp'] == board['timestamp'] and ocr_entry['path'] == board['path']:
                    ocr_img = ocr_entry.get('ocr_img')
                    break
        
        # If no OCR image was found in pre-optimized data, load from the file path
        if ocr_img is None:
            # Get the image path - handle the path correctly
            image_path = board['path']
            
            # If the path is not absolute, make it relative to the JSON directory
            if not os.path.isabs(image_path):
                # Handle different path formats - use only the basename if it's likely a duplicate path
                if 'boards/' in image_path or os.path.basename(json_dir) in image_path:
                    # Strip redundant directory prefixes
                    image_basename = os.path.basename(image_path)
                    image_path = os.path.join(json_dir, image_basename)
                else:
                    # Simply join with the JSON directory
                    image_path = os.path.join(json_dir, image_path)
            
            # Skip if the image doesn't exist
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_path} does not exist. Trying alternative paths...")
                
                # Try a few common alternative paths
                alternatives = [
                    os.path.join(json_dir, os.path.basename(image_path)),
                    os.path.join(os.path.dirname(json_dir), os.path.basename(image_path)),
                    image_path.replace(f"{os.path.basename(json_dir)}/", "")
                ]
                
                for alt_path in alternatives:
                    if os.path.exists(alt_path):
                        print(f"Found alternative path: {alt_path}")
                        image_path = alt_path
                        break
                else:
                    print(f"Error: Could not locate image {os.path.basename(image_path)} after trying alternatives. Skipping.")
                    # Save board entry with error message instead of skipping completely
                    board['text'] = "%error: image not found"
                    with open(json_path, 'w') as f:
                        json.dump(boards_data, f, indent=2)
                    continue
            
            # We'll use the file path since we don't have a pre-optimized image
            ocr_input = image_path
        else:
            # We'll use the pre-optimized image
            ocr_input = ocr_img
        
        # Process the image and get the LaTeX text
        processed_count += 1
        print(f"Processing image {processed_count}/{len(entries_to_process)} (entry {i+1}/{total_entries}): {os.path.basename(board['path'])}")
        
        try:
            # Use the multi-stage OCR process
            latex_text = multi_stage_ocr_process(ocr_input)
            
            # Add the text field to the board entry
            board['text'] = latex_text
            
            # Save the updated JSON file after each successful API call
            # This ensures we don't lose progress if an error occurs later
            with open(json_path, 'w') as f:
                json.dump(boards_data, f, indent=2)
            print(f"Saved progress to {json_path} after processing image {processed_count}/{len(entries_to_process)}")
        except Exception as e:
            # If there's an error processing this image, mark it as an error but don't crash
            error_msg = f"%error: {str(e)}"
            print(f"Error processing image: {error_msg}")
            
            # Mark the entry with the error
            board['text'] = error_msg
            
            # Save the progress so far
            with open(json_path, 'w') as f:
                json.dump(boards_data, f, indent=2)
            print(f"Saved error state to {json_path}")
    
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
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug mode with visualization of preprocessing steps")
    parser.add_argument("--force-optimize", action="store_true",
                      help="Force re-optimization of images even if OCR data exists")
    args = parser.parse_args()
    
    # Check if the input is a JSON file or an image
    if args.input_path.endswith('.json'):
        process_boards_json(args.input_path)
    else:
        # Process a single image
        if args.debug:
            # Show preprocessing steps in debug mode
            img = cv2.imread(args.input_path)
            img_proc = optimize_image_for_ocr(args.input_path)
            
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title("Original Image")
            plt.axis("off")
            
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB))
            plt.title("OCR-Optimized Image")
            plt.axis("off")
            
            plt.tight_layout()
            plt.show()
        
        # Run the multi-stage OCR process
        latex_output = multi_stage_ocr_process(args.input_path)
        print("\n--- LaTeX Output ---\n")
        print(latex_output)

if __name__ == "__main__":
    main()
