import os
import json
import time
import subprocess
import requests
from dotenv import load_dotenv

# Load API key
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

# Config
MODEL = "qwen/qwen2.5-vl-72b-instruct:free"  # You can change this
OUTPUT_DIR = "lecture_notes_tex"
LECTURE_FILE = "lecture.json"
MASTER_FILE = "master.tex"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load lecture segments
with open(LECTURE_FILE, "r") as f:
    lecture_data = json.load(f)
segments = lecture_data["segments"]

def create_prompt(spoken, written_list):
    board_texts = "\n\n".join([f"[Board at {w['timestamp']}]:\n{w['text']}" for w in written_list])

    template_description = r"""
The LaTeX document uses:
- \documentclass{article}
- \input{shortcuts.tex} for custom macros.
- \usepackage{quiver} for diagrams.
- AMS packages like amsmath, amssymb are available.

Format notes with sections, definitions, theorems, remarks, examples.
Use quiver diagrams where appropriate.
"""

    prompt = rf"""
You are writing professional LaTeX lecture notes.

{template_description}

Spoken lecture:
---
{spoken}
---

Blackboard extracts:
---
{board_texts if board_texts else '(No blackboard content)'}
---

Output pure LaTeX only, suitable for compiling with the above setup.
"""
    return prompt.strip()

def send_to_openrouter(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://yourdomain.com",
        "X-Title": "LectureNoteGeneratorTex"
    }
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a highly skilled LaTeX math lecture note-taker."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(OPENROUTER_ENDPOINT, headers=headers, json=body)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# Main processing loop
all_blocks = []
for idx, segment in enumerate(segments):
    output_path = os.path.join(OUTPUT_DIR, f"block_{idx:04d}.tex")

    if os.path.exists(output_path):
        print(f"Skipping block {idx} (already exists).")
        all_blocks.append(output_path)
        continue

    print(f"Processing block {idx} ({segment['start_time']} - {segment['end_time']})...")

    prompt = create_prompt(segment["spoken_content"], segment.get("written_content", []))

    try:
        notes = send_to_openrouter(prompt)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"% Notes for {segment['start_time']} - {segment['end_time']}\n\n")
            f.write(notes)
        all_blocks.append(output_path)
        print(f"Saved block {idx}.")
    except Exception as e:
        print(f"Error on block {idx}: {e}")
        print("Sleeping for 10 seconds before retry...")
        time.sleep(10)

# Create master.tex
print("\nGenerating master.tex...")
with open(MASTER_FILE, "w", encoding="utf-8") as f:
    f.write(r"""% Auto-generated master file
\documentclass{article}
\makeatletter
\def\input@path{{tex_resources/}}
\makeatother
\input{shortcuts.tex}
\usepackage{quiver}
\begin{document}

""")
    for block_path in sorted(all_blocks):
        relative_path = os.path.relpath(block_path, start=os.path.dirname(MASTER_FILE))
        latex_path = relative_path.replace("\\", "/")
        f.write(f"\input{{{latex_path}}}\n")
    f.write(r"""

\end{document}
""")
print("Master file created!")

# Optional: Compile the master.tex
print("Compiling master.tex with pdflatex...")
try:
    subprocess.run(["pdflatex", MASTER_FILE], check=True)
    subprocess.run(["pdflatex", MASTER_FILE], check=True)  # Twice for TOC/refs
except Exception as e:
    print(f"Warning: LaTeX compilation failed: {e}")

# Optional: Cleanup auxiliary files
aux_files = [".aux", ".log", ".out", ".toc"]
for ext in aux_files:
    try:
        os.remove(MASTER_FILE.replace(".tex", ext))
    except FileNotFoundError:
        pass

print("\nAll done. Master PDF should be ready!")
