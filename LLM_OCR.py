import argparse
import base64
import cv2
import requests
import numpy as np
import os
import time
import re
import json
from pathlib import Path
from dotenv import load_dotenv
import sys

# Load environment variables
print("Loading environment variables...")
load_dotenv(verbose=True)

# Get API key
API_KEY = os.getenv("OPENROUTER_API_KEY")
print(f"API Key loaded: {'Yes (length: ' + str(len(API_KEY)) + ')' if API_KEY else 'No'}")

if not API_KEY:
    print("ERROR: OpenRouter API key not found in environment variables.")
    print("Please add your API key to the .env file as: OPENROUTER_API_KEY=your_key_here")
    print("Get your OpenRouter API key from: https://openrouter.ai/keys")
    sys.exit(1)
else:
    print("API key loaded successfully!")

# OpenRouter endpoint
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# Fix: Use the API key directly in the Authorization header without any transformation
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "HTTP-Referer": "https://github.com/DaveArcher18/LecToNotes",
    "X-Title": "LecToNotes OCR", 
    "Content-Type": "application/json"
}

# Print header info for debugging (masking the actual key)
key_preview = API_KEY[:5] + "..." + API_KEY[-4:] if len(API_KEY) > 9 else "***"
print(f"Authorization header: Bearer {key_preview}")

# Models
OCR_MODEL = "meta-llama/llama-4-maverick:free"
REFINER_MODEL = "deepseek/deepseek-prover-v2:free"
VALIDATOR_MODEL = "deepseek/deepseek-prover-v2:free"
THROTTLETIME = 0.5  # Small delay to avoid rate limits

# Common mathematical structure patterns to validate
MATH_PATTERNS = {
    "matrix": r"\\begin{(pmatrix|bmatrix|vmatrix|matrix)}(.*?)\\end{\1}",
    "align": r"\\begin{align}(.*?)\\end{align}",
    "equation": r"\\begin{equation}(.*?)\\end{equation}",
    "cases": r"\\begin{cases}(.*?)\\end{cases}",
    "array": r"\\begin{array}(.*?)\\end{array}",
}

def call_openrouter_api(messages, model, temperature=0.0, max_tokens=None, response_format=None):
    """Call OpenRouter API with the specified parameters."""
    # Prepare payload
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
        
    if response_format is not None:
        payload["response_format"] = response_format
    
    print(f"Calling OpenRouter API with model: {model}")
    
    # Make the request with fixed headers
    try:
        # Debug the actual headers being sent (without showing full API key)
        masked_headers = HEADERS.copy()
        if "Authorization" in masked_headers:
            key = masked_headers["Authorization"][7:]  # Remove "Bearer " prefix
            masked_headers["Authorization"] = f"Bearer {key[:5]}...{key[-4:]}" if len(key) > 9 else "Bearer ***"
        
        print(f"Request headers: {masked_headers}")
        print(f"Endpoint: {ENDPOINT}")
        
        response = requests.post(
            ENDPOINT, 
            headers=HEADERS,  # Use the global headers directly 
            json=payload,
            timeout=60
        )
        
        if response.status_code == 401:
            print(f"Authentication error: Invalid or missing API key.")
            print(f"Error details: {response.text}")
            print("Please check that your API key is correctly formatted and valid.")
            sys.exit(1)
        
        if response.status_code != 200:
            print(f"OpenRouter API error (HTTP {response.status_code}): {response.text}")
            return f"ERROR: API request failed with status code {response.status_code}"
        
        # Process successful response
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
        
    except Exception as e:
        print(f"Error during API call: {str(e)}")
        return f"ERROR: Failed to get response from the API. {str(e)}"

def encode_image_to_data_uri(image_path):
    """Convert image file to base64 data URI."""
    # Read image directly from file
    with open(image_path, "rb") as img_file:
        b64 = base64.b64encode(img_file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def find_relevant_transcript_context(timestamp, transcript_data):
    """
    Given a timestamp from a board, find relevant context from the transcript.
    Returns both relevant transcript segment and summaries.
    """
    if not transcript_data:
        return None, None
    
    # Convert timestamp to seconds for easier comparison
    if isinstance(timestamp, str) and '_' in timestamp:
        # Format: "HH_MM_SS"
        parts = timestamp.split('_')
        if len(parts) >= 3:
            hour = int(parts[0])
            minute = int(parts[1])
            second = int(parts[2]) if parts[2] else 0
            timestamp_seconds = hour * 3600 + minute * 60 + second
        else:
            return None, None
    else:
        return None, None
    
    # Find the relevant segment in the transcript
    current_segment = None
    previous_segment = None
    next_segment = None
    all_summaries = []
    next_segment_start_seconds = float('inf')
    
    for segment in transcript_data:
        # Extract start and end times
        start_parts = segment["start"].split('_')
        end_parts = segment["end"].split('_')
        
        start_seconds = int(start_parts[0]) * 3600 + int(start_parts[1]) * 60 + int(start_parts[2])
        end_seconds = int(end_parts[0]) * 3600 + int(end_parts[1]) * 60 + int(end_parts[2])
        
        # Add summary to all summaries if available
        if "summary" in segment and segment["summary"].strip():
            all_summaries.append(segment["summary"])
        
        # Check if timestamp falls within this segment
        if start_seconds <= timestamp_seconds < end_seconds:
            current_segment = segment
        elif end_seconds <= timestamp_seconds and (next_segment is None or end_seconds > previous_segment.get("end_seconds", 0) if previous_segment else 0):
            previous_segment = segment
            previous_segment["end_seconds"] = end_seconds
        elif timestamp_seconds < start_seconds and (next_segment is None or start_seconds < next_segment_start_seconds):
            next_segment = segment
            next_segment_start_seconds = start_seconds
    
    # Create context from the segments
    context = ""
    if current_segment:
        context = f"Current segment ({current_segment['start']} to {current_segment['end']}):\n{current_segment['content']}\n\n"
    
    if previous_segment:
        context += f"Previous segment ({previous_segment['start']} to {previous_segment['end']}):\n{previous_segment['content']}\n\n"
    
    if next_segment:
        context += f"Next segment ({next_segment['start']} to {next_segment['end']}):\n{next_segment['content']}"
    
    return context, all_summaries

def extract_latex_with_ocr(image_data_uri, transcript_context=None, all_past_summaries=None):
    """First pass: Extract raw content from image using Llama 4 Maverick with transcript context and all past summaries."""
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
   
5. IMPORTANT FORMATTING RESTRICTIONS:
   - DO NOT include \\documentclass, \\usepackage, \\begin{document}, or \\end{document}
   - The output will be directly displayed on a website, not compiled as a LaTeX document
   - DO NOT include any LaTeX preamble or document setup commands
   - Focus ONLY on the mathematical content itself

IMPORTANT: Focus on accurately transcribing the mathematical content rather than interpreting or explaining it. Preserve all mathematical symbols exactly as they appear.
"""

    user_content = [
        {"type": "text", "text": "Transcribe the mathematical content from this blackboard image into precise LaTeX/Markdown. The output will be displayed directly on a website, so do NOT include any LaTeX document setup commands like \\documentclass, \\usepackage, or \\begin{document}."},
        {"type": "image_url", "image_url": {"url": image_data_uri}}
    ]
    
    # Add transcript context if available
    if transcript_context:
        context_text = f"""
I am providing you with relevant transcript segments from the lecture to help you understand the context of what's being discussed on the blackboard. 
This should help you accurately interpret the mathematical notation and terminology:

{transcript_context}
        """
        user_content[0]["text"] += "\n\n" + context_text
        
    # Add ALL past lecture summaries if available
    if all_past_summaries and len(all_past_summaries) > 0:
        summaries_text = "\nLECTURE HISTORICAL CONTEXT (all previous lecture segments):\n"
        for i, summary in enumerate(all_past_summaries):
            summaries_text += f"Summary {i+1}:\n{summary}\n\n"
        user_content[0]["text"] += "\n" + summaries_text

    ocr_messages = [
        {"role": "system", "content": ocr_system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    return call_openrouter_api(ocr_messages, OCR_MODEL, temperature=0.1, max_tokens=1500)

def validate_and_structure_latex(raw_text, transcript_context=None, all_past_summaries=None):
    """Combined function to structure and validate LaTeX content"""
    
    # Include context in the prompt if available
    context_section = ""
    if transcript_context:
        context_section = f"""
LECTURE CONTEXT:
{transcript_context}

Use this context to help you understand the mathematical terminology and notation while structuring and validating the content.
"""
    
    # Include summaries in the prompt if available
    summaries_section = ""
    if all_past_summaries and len(all_past_summaries) > 0:
        summaries_section = "\nLECTURE HISTORICAL CONTEXT:\n"
        for i, summary in enumerate(all_past_summaries[:5]):  # Use first 5 summaries for context
            summaries_section += f"Summary {i+1}:\n{summary}\n\n"
    
    combined_prompt = f"""You are an expert LaTeX formatter and validator specializing in advanced mathematics. Your task is to structure and validate the raw mathematical content extracted from a blackboard.

Raw extracted content:
```
{raw_text}
```
{context_section}
{summaries_section}

Please transform this into well-structured, valid web-compatible LaTeX/Markdown by:

1. WEBSITE FORMATTING REQUIREMENTS:
   - DO NOT include \\documentclass, \\usepackage, \\begin document, or \\end document
   - DO NOT include any LaTeX preamble or document setup
   - The output will be displayed on a website, not compiled as a LaTeX document
   - Focus ONLY on the mathematical content itself

2. Mathematical Structure:
   - Identifying mathematical structures (equations, matrices, etc.) and ensuring they use the correct LaTeX environments
   - Ensuring all mathematical expressions are properly delimited with $ or $$
   - Using appropriate environments like align, matrix, cases, etc.
   - Ensuring all environments are properly opened and closed
   - Ensuring proper nesting of delimiters and environments
   - Fixing common OCR errors in mathematical notation

3. Syntax Validation:
   - Convert all instances of \\2 to {2} for squared terms
   - Convert _2 to _{2} for subscripts
   - Fix any unescaped special characters like % _ & # etc.
   - Replace \\section{{}} and \\subsection{{}} with markdown ## and ### respectively
   - Ensure all \\begin{{environment}} have matching \\end{{environment}}
   - Fix any mismatched $$...$ or $...$$
   - Use proper mathematical notation for special symbols

4. Content Formatting:
   - Use the context provided to correctly identify mathematical symbols and terminology
   - Make sure to preserve mathematical meaning while improving formatting
   - Organize content in a logical flow
   - Use Markdown headings (## for sections, ### for subsections) for clear structure
   - Insert blank lines between different sections/topics
   - For inline math, use $...$ notation
   - For display math, use $$...$$ notation

Return ONLY the structured and validated LaTeX/Markdown content with no explanation or commentary.
"""

    combined_messages = [
        {"role": "system", "content": "You are an expert LaTeX validator and formatter specializing in advanced mathematics."},
        {"role": "user", "content": combined_prompt}
    ]
    
    return call_openrouter_api(combined_messages, REFINER_MODEL, temperature=0.1)

def ocr_process_image(image_path, transcript_data=None, timestamp=None, all_past_summaries=None):
    """
    Simplified OCR process that uses pre-processed images directly.
    """
    # Convert image to data URI
    data_uri = encode_image_to_data_uri(image_path)
    
    # Get relevant transcript context if available
    transcript_context = None
    if transcript_data and timestamp:
        transcript_context, segment_summaries = find_relevant_transcript_context(timestamp, transcript_data)
        if transcript_context:
            print("Found relevant transcript context for this board.")
    
    # First pass OCR with Llama 4 Maverick
    print("Extracting raw content...")
    raw_text = extract_latex_with_ocr(data_uri, transcript_context, all_past_summaries)
    
    # Skip further processing if illegible
    if raw_text.strip() == "%illegible":
        return "%illegible"
    
    # Combined structure and validation with DeepSeek Prover
    print("Structuring and validating content...")
    validated_text = validate_and_structure_latex(raw_text, transcript_context, all_past_summaries)
    
    # Final postprocessing
    print("Final postprocessing...")
    final_text = enhanced_postprocess_text(validated_text)
    
    return final_text

def enhanced_postprocess_text(text):
    """Enhanced postprocessing with specific rules for mathematical notation."""
    # Remove LaTeX document setup commands
    text = re.sub(r'\\documentclass.*?(\n|$)', '', text)
    text = re.sub(r'\\usepackage.*?(\n|$)', '', text)
    text = re.sub(r'\\begin\{document\}.*?(\n|$)', '', text)
    text = re.sub(r'\\end\{document\}.*?(\n|$)', '', text)
    
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
    
    # Clean up any remaining document class/package residue
    text = re.sub(r'\$\$ \\begin\{document\}', r'$$', text)
    text = re.sub(r'\\end\{document\} \$\$', r'$$', text)
    
    return text.strip()

def process_boards_json(json_path, transcript_path=None):
    """Process all boards in a JSON file with optional transcript context."""
    # Load transcript data if provided
    transcript_data = None
    if transcript_path:
        print(f"Loading transcript data from {transcript_path}...")
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        print(f"Loaded {len(transcript_data)} transcript segments.")
    
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
    
    print(f"Processing {len(entries_to_process)} boards...")
    
    # Process each board entry that doesn't already have text
    processed_count = 0
    for i, board in enumerate(boards_data):
        # Skip if this entry already has text
        if 'text' in board:
            continue
            
        # Get the image path from the board entry
        image_basename = os.path.basename(board['path'])
        
        # Directly construct the path to the OCR-processed image
        ocr_basename = image_basename.replace('board_', 'ocr_')
        ocr_image_path = os.path.join(json_dir, 'ocr', ocr_basename)
        
        # Check if the OCR image exists
        if not os.path.exists(ocr_image_path):
            print(f"Warning: OCR image not found at {ocr_image_path}")
            # Look in alternative locations
            alt_paths = [
                os.path.join(json_dir, 'boards', 'ocr', ocr_basename),
                os.path.join(os.path.dirname(json_dir), 'boards', 'ocr', ocr_basename)
            ]
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    ocr_image_path = alt_path
                    print(f"Found OCR image at {ocr_image_path}")
                    break
            else:
                print(f"Error: OCR image not found for {ocr_basename}. Skipping.")
                board['text'] = "%error: OCR image not found"
                continue
        
        # Process the image
        processed_count += 1
        timestamp = board.get('timestamp')
        print(f"Processing image {processed_count}/{len(entries_to_process)}: {image_basename} ({timestamp})")
        
        # Get all transcript summaries up to this timestamp
        all_past_summaries = []
        
        if transcript_data and timestamp:
            # Parse the timestamp (expected format: "HH_MM_SS")
            if isinstance(timestamp, str) and '_' in timestamp:
                parts = timestamp.split('_')
                if len(parts) >= 3:
                    hour = int(parts[0])
                    minute = int(parts[1])
                    second = int(parts[2]) if parts[2] else 0
                    timestamp_seconds = hour * 3600 + minute * 60 + second
                    
                    # Collect all summaries from segments before this timestamp
                    for segment in transcript_data:
                        # Extract end time of segment
                        end_parts = segment["end"].split('_')
                        end_seconds = int(end_parts[0]) * 3600 + int(end_parts[1]) * 60 + int(end_parts[2])
                        
                        # If segment ends before current timestamp and has a summary, include it
                        if end_seconds <= timestamp_seconds and "summary" in segment and segment["summary"].strip():
                            all_past_summaries.append(segment["summary"])
        
        print(f"Using {len(all_past_summaries)} historical summaries for context")
        
        # Process the OCR image
        latex_text = ocr_process_image(ocr_image_path, transcript_data, timestamp, all_past_summaries)
        
        # Add the text field to the board entry
        board['text'] = latex_text
        
        # Save the updated JSON file after each successful API call
        with open(json_path, 'w') as f:
            json.dump(boards_data, f, indent=2)
        
        # Add a delay to avoid rate limiting
        if THROTTLETIME > 0 and processed_count < len(entries_to_process):
            time.sleep(THROTTLETIME)
    
    print(f"✅ Completed processing {processed_count} boards!")
    return boards_data

def main():
    parser = argparse.ArgumentParser(
        description="Process blackboard images and transcribe content to LaTeX."
    )
    parser.add_argument("input_path", 
                      help="Path to either a single image or a boards.json file")
    parser.add_argument("--transcript", 
                      help="Path to the transcript.json file for context enhancement")
    parser.add_argument("--ocr-dir",
                      help="Directory containing OCR-processed images (default: 'ocr' relative to input)")
    args = parser.parse_args()
    
    # Check if input path exists
    if not os.path.exists(args.input_path):
        print(f"ERROR: Input path '{args.input_path}' not found.")
        sys.exit(1)
    
    # Check if transcript exists if provided
    if args.transcript and not os.path.exists(args.transcript):
        print(f"WARNING: Transcript file '{args.transcript}' not found.")
        args.transcript = None
    
    # Test the connection with a simple API call
    print("Testing API connection...")
    test_messages = [
        {"role": "system", "content": "Test message to verify API connection."},
        {"role": "user", "content": "This is a test. Please respond with 'API connection successful.'"}
    ]
    response = call_openrouter_api(test_messages, OCR_MODEL, temperature=0.0)
    
    if "API connection successful" in response:
        print("API connection confirmed.")
    
    # Check if the input is a JSON file or an image
    if args.input_path.endswith('.json'):
        process_boards_json(args.input_path, args.transcript)
    else:
        # Process a single image file
        print(f"Processing single image: {args.input_path}")
        
        # Determine timestamp from filename if possible
        timestamp = None
        filename = os.path.basename(args.input_path)
        if filename.startswith("board_") and "_" in filename:
            parts = filename.split("_")
            if len(parts) >= 4:
                timestamp = f"{parts[1]}_{parts[2]}_{parts[3].split('.')[0]}"
                print(f"Extracted timestamp: {timestamp}")
        
        # Process the image
        latex_output = ocr_process_image(args.input_path)
        print("\n--- LaTeX Output ---\n")
        print(latex_output)

if __name__ == "__main__":
    main()
