import os
import re
import json
import time
from datetime import datetime
import requests
from dotenv import load_dotenv

# Load API key
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

# Config
MODEL = "thudm/glm-z1-32b:free"
# meta-llama/llama-4-maverick:free
# thudm/glm-z1-32b:free
# qwen/qwen2.5-vl-72b-instruct:free
LECTURE_FILE = "lecture.json"
MASTER_FILE = "master.tex"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# Today's date
today_str = datetime.today().strftime('%B %d, %Y')

# Load lecture segments
with open(LECTURE_FILE, "r") as f:
    lecture_data = json.load(f)
segments = lecture_data["segments"]

def create_prompt(spoken, written_list):
    board_texts = "\n\n".join([f"[Board at {w['timestamp']}]:\n{w['text']}" for w in written_list])

    template_description = r"""
The LaTeX document uses:
- \documentclass{amsart}
- Many math packages, TikZ, quiver, cleveref, etc.
- Theorems, definitions, examples, exercises, etc., are allowed.

IMPORTANT:
You must NOT include \documentclass{}, \usepackage{}, \begin{document}, or \end{document}.
Only produce clean body content suitable for direct insertion into the main file.

Use sections, definitions, theorems, examples, remarks appropriately.
Draw diagrams using quiver if necessary.
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

Please output only clean LaTeX body content, without document wrappers.
"""
    return prompt.strip()

def sanitize_latex_content(content: str) -> str:
    forbidden_patterns = [
        r"\\documentclass\{.*?\}",
        r"\\usepackage\{.*?\}",
        r"\\begin\{document\}",
        r"\\end\{document\}"
    ]
    for pattern in forbidden_patterns:
        content = re.sub(pattern, "", content, flags=re.IGNORECASE)
    return content.strip()

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

def write_preamble(f):
    f.write(rf"""% Auto-generated master file
\documentclass{{amsart}}
\usepackage[margin=1.5in]{{geometry}}
\usepackage{{amsmath}}
\usepackage{{tcolorbox}}
\usepackage{{amssymb}}
\usepackage{{amsthm}}
\usepackage{{lastpage}}
\usepackage{{fancyhdr}}
\usepackage{{accents}}
\usepackage{{hyperref}}
\usepackage{{xcolor}}
\usepackage{{color}}
\usepackage[bbgreekl]{{mathbbol}}
\DeclareSymbolFontAlphabet{{\mathbb}}{{AMSb}}
\DeclareSymbolFontAlphabet{{\mathbbl}}{{bbold}}
\input{{shortcuts.tex}}
\setlength{{\headheight}}{{40pt}}

\usepackage{{amsmath, amssymb, tikz, amsthm, csquotes, multicol, footnote, tablefootnote, biblatex, wrapfig, float, quiver, mathrsfs, cleveref, enumitem, stmaryrd, marginnote, todonotes, euscript}}
\addbibresource{{refs.bib}}
\theoremstyle{{definition}}
\newtheorem{{theorem}}{{Theorem}}[section]
\newtheorem{{lemma}}[theorem]{{Lemma}}
\newtheorem{{corollary}}[theorem]{{Corollary}}
\newtheorem{{exercise}}[theorem]{{Exercise}}
\newtheorem{{question}}[theorem]{{Question}}
\newtheorem{{example}}[theorem]{{Example}}
\newtheorem{{proposition}}[theorem]{{Proposition}}
\newtheorem{{conjecture}}[theorem]{{Conjecture}}
\newtheorem{{remark}}[theorem]{{Remark}}
\newtheorem{{definition}}[theorem]{{Definition}}
\numberwithin{{equation}}{{section}}
\setuptodonotes{{color=blue!20, size=tiny}}

\newenvironment{{solution}}
  {{\renewcommand\qedsymbol{{$\\blacksquare$}}
  \begin{{proof}}[Solution]}}
  {{\end{{proof}}}}
\renewcommand\qedsymbol{{$\\blacksquare$}}

\pagestyle{{fancy}}
\fancyhf{{}}
\lhead{{Lecture Notes}}
\chead{{Generated on {today_str}}}
\rhead{{\thepage}}

\begin{{document}}

""")

def append_notes(f):
    for idx, segment in enumerate(segments):
        print(f"Processing block {idx} ({segment['start_time']} - {segment['end_time']})...")
        prompt = create_prompt(segment["spoken_content"], segment.get("written_content", []))

        try:
            notes = send_to_openrouter(prompt)
            notes = sanitize_latex_content(notes)
            f.write(f"% Block {idx}: {segment['start_time']} - {segment['end_time']}\n\n")
            f.write(notes + "\n\n")
            print(f"Added block {idx}.")
        except Exception as e:
            print(f"Error on block {idx}: {e}")
            print("Sleeping for 10 seconds before retry...")
            time.sleep(10)

def finalize_document(f):
    f.write("\n\\end{document}\n")

# Main execution
if __name__ == "__main__":
    with open(MASTER_FILE, "w", encoding="utf-8") as f:
        write_preamble(f)
        append_notes(f)
        finalize_document(f)

    print("\n✅ master.tex created successfully!")