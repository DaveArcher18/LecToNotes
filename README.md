# LecToNotes

LecToNotes is a software package for extracting text resources from technical research mathematics lecture videos. It processes both the spoken audio and blackboard content, converting them into structured JSON data for easy downstream display and analysis.

## Overview

The system extracts two primary types of content from lecture videos:

1. **Spoken Transcript**: The audio is transcribed into text using either local Whisper or Groq's API.
2. **Blackboard Content**: Screenshots of the blackboard are captured, deduplicated, and converted to LaTeX using an LLM-based OCR system.

These resources are then merged into a unified JSON structure that organizes the lecture content into time-segmented blocks.

## System Components

### Core Scripts

- `get_boards.py`: Extracts and deduplicates blackboard images from lecture videos
- `get_transcript.py`: Transcribes audio from lecture videos using Whisper or Groq
- `LLM_OCR.py`: Converts blackboard images to LaTeX using LLM-based OCR
- `merge/unified_merge.py`: Merges board content and transcript into a unified structure

### Output Files

- `boards.json`: Contains timestamped blackboard images and their LaTeX transcriptions
- `transcript.json`: Contains timestamped segments of the lecture transcript
- `merge/lecture.json`: The final unified representation of the lecture content

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/LecToNotes.git
cd LecToNotes

# Install dependencies
uv pip install -r requirements.txt
```

## Usage

### Complete Pipeline

The complete pipeline can be run using the provided Makefile:

```bash
# Process a lecture video (extracts boards, transcript, and merges them)
make process VIDEO=path/to/lecture.mp4 TITLE="Lecture Title" DATE=YYYY-MM-DD
```

## Makefile Guide

The project includes a comprehensive Makefile that automates the entire lecture processing pipeline. This makes it easy to process videos with a single command or run individual steps as needed.

### Makefile Structure

The Makefile defines several targets that correspond to different stages of the processing pipeline:

- `process`: Runs the complete pipeline (transcript extraction, board extraction, OCR, and merging)
- `extract_transcript`: Extracts the transcript from the video
- `extract_boards`: Extracts blackboard images from the video
- `ocr_boards`: Processes the extracted board images with OCR
- `merge`: Merges the transcript and board content into a unified structure
- `clean`: Removes all generated files
- `help`: Displays help information about the Makefile

### Key Parameters

The Makefile accepts several parameters that control the processing:

- `VIDEO`: Path to the input video file (required)
- `TITLE`: Title of the lecture (default: "Untitled Lecture")
- `DATE`: Date of the lecture in YYYY-MM-DD format (optional)
- `INTERVAL`: Time interval for segmentation in seconds (default: 300)
- `USE_GROQ`: Whether to use Groq API for transcription (true/false, default: false)

### Output Structure

All output files are stored in a directory named `<TITLE> OUTPUT`, with the following structure:

```
<TITLE> OUTPUT/
├── transcript.json         # Extracted transcript
├── boards/                 # Directory containing board images
│   ├── board_*.jpg         # Extracted board images
│   └── boards.json         # Metadata for board images
└── merge/                  # Directory containing merged content
    └── lecture.json        # Final unified lecture content
```

### Example Usage

#### Basic Usage

```bash
# Process a local video file
make process VIDEO=/path/to/lecture.mp4 TITLE="Linear Algebra Lecture 1"
```

#### Processing a YouTube Video

```bash
# Process a YouTube video
make process VIDEO="https://youtube.com/watch?v=VIDEO_ID" TITLE="Calculus Lecture 2" DATE="2023-09-15"
```

#### Using Groq API for Transcription

```bash
# Process a video using Groq API for transcription
make process VIDEO=/path/to/lecture.mp4 TITLE="Quantum Mechanics" USE_GROQ=true
```

#### Customizing Segment Interval

```bash
# Process a video with custom segment interval (10 minutes)
make process VIDEO=/path/to/lecture.mp4 TITLE="Statistics" INTERVAL=600
```

#### Running Individual Steps

```bash
# Extract transcript only
make extract_transcript VIDEO=/path/to/lecture.mp4 TITLE="Physics Lecture"

# Extract board images only
make extract_boards VIDEO=/path/to/lecture.mp4 TITLE="Chemistry Lecture"

# Process board images with OCR (after extraction)
make ocr_boards TITLE="Chemistry Lecture"

# Merge transcript and boards (after extraction and OCR)
make merge TITLE="Chemistry Lecture" DATE="2023-10-20"
```

#### Cleaning Generated Files

```bash
# Remove all generated files for a specific lecture
make clean TITLE="Linear Algebra Lecture 1"
```

For more information about the Makefile, run `make help` to display the built-in help message.

### Individual Components

Alternatively, you can run each component separately:

> **Important Note**: When using YouTube URLs as input, you must run `get_transcript.py` first, as it handles YouTube video downloading. The downloaded video can then be used as input for `get_boards.py`.

#### 1. Generate Transcript

```bash
# Using local Whisper (with local mp4 file)
python get_transcript.py path/to/lecture.mp4 --out transcript.json

# Using local Whisper (with YouTube URL)
python get_transcript.py https://youtube.com/watch?v=VIDEO_ID --out transcript.json

# Using Groq API
python get_transcript.py path/to/lecture.mp4 --use-groq --out transcript.json
```

#### 2. Extract Blackboard Images

```bash
python get_boards.py -i path/to/lecture.mp4 -o boards/
```

#### 3. Process Blackboard Images with OCR

```bash
python LLM_OCR.py boards/boards.json
```

#### 4. Merge Boards and Transcript

```bash
python merge/unified_merge.py -b boards/boards.json -t transcript.json -o merge/lecture.json --title "Lecture Title" --date YYYY-MM-DD
```

## Environment Variables

Create a `.env` file with the following API keys:

```
GROQ_API_KEY=your_groq_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

## Output Format

### boards.json

```json
[
  {
    "timestamp": "00_03_24",
    "path": "boards/board_00_03_24_000.jpg",
    "text": "\\begin{center}\nNo Lecture next two weeks...\\end{center}"
  },
  ...
]
```

### transcript.json

```json
[
  {
    "start": "00_00_00",
    "end": "00_05_00",
    "content": "Welcome to today's lecture on Habiro Cohomology..."
  },
  ...
]
```

### lecture.json

```json
{
  "lecture": "Habiro Cohomology – Lecture 1",
  "date": "2025-04-24",
  "segments": [
    {
      "start_time": "00:00:00",
      "end_time": "00:05:00",
      "spoken_content": "Welcome to today's lecture...",
      "written_content": [
        {
          "timestamp": "00_03_24",
          "path": "boards/board_00_03_24_000.jpg",
          "text": "\\begin{center}\nNo Lecture next two weeks...\\end{center}"
        }
      ]
    },
    ...
  ]
}
```

## Dependencies

See `requirements.txt` for a complete list of dependencies.

## License

[MIT](LICENSE)