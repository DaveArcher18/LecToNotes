# LecToNotes Processing Flow

```mermaid
graph TD
    A[Input: Lecture Video] --> C[get_transcript.py]
    C --> E[transcript.json]
    
    %% For YouTube videos, get_transcript.py downloads the video first
    C -- "If YouTube URL" --> V[Downloaded Video]
    V --> B[get_boards.py]
    %% For local videos, use directly
    A -- "If local file" --> B[get_boards.py]
    
    B --> D[boards/boards.json]
    
    D --> F[LLM_OCR.py]
    F --> G[Updated boards.json with LaTeX]
    
    G --> H[merge/unified_merge.py]
    E --> H
    
    H --> I[merge/lecture.json]
    
    subgraph "1. Content Extraction"
        C[get_transcript.py]
        B[get_boards.py]
    end
    
    subgraph "2. OCR Processing"
        F[LLM_OCR.py]
    end
    
    subgraph "3. Content Merging"
        H[merge/unified_merge.py]
    end
    
    subgraph "Output Files"
        D[boards/boards.json]
        E[transcript.json]
        G[Updated boards.json with LaTeX]
        I[merge/lecture.json]
    end
    
    classDef process fill:#f9f,stroke:#333,stroke-width:2px;
    classDef data fill:#bbf,stroke:#333,stroke-width:1px;
    classDef input fill:#bfb,stroke:#333,stroke-width:1px;
    classDef output fill:#fbb,stroke:#333,stroke-width:1px;
    
    class A input;
    class B,C,F,H process;
    class D,E,G data;
    class I output;
```

## Processing Steps

1. **Content Extraction**
   - `get_boards.py`: Extracts blackboard images from the video, deduplicates them, and saves metadata to `boards.json`
   - `get_transcript.py`: Transcribes the audio from the video using either local Whisper or Groq API and saves to `transcript.json`

2. **OCR Processing**
   - `LLM_OCR.py`: Processes each blackboard image using an LLM-based OCR system to convert the content to LaTeX and updates `boards.json`

3. **Content Merging**
   - `merge/unified_merge.py`: Combines the blackboard content and transcript into a unified structure organized by time segments in `lecture.json`

## Command Flow

```bash
# Complete pipeline using Makefile
make process VIDEO=path/to/lecture.mp4 TITLE="Lecture Title" DATE=YYYY-MM-DD

# Or run individual steps manually:

# For YouTube videos:
# 1. Extract transcript (this will download the video)
python get_transcript.py https://youtube.com/watch?v=VIDEO_ID --out transcript.json
# Then use the downloaded video for the next steps

# For local video files:
# 1. Extract transcript
python get_transcript.py path/to/lecture.mp4 --out transcript.json
# OR with Groq API
python get_transcript.py path/to/lecture.mp4 --use-groq --out transcript.json

# 2. Extract board images
python get_boards.py -i path/to/lecture.mp4 -o boards/

# 3. Process board images with OCR
python LLM_OCR.py boards/boards.json

# 4. Merge boards and transcript
python merge/unified_merge.py -b boards/boards.json -t transcript.json -o merge/lecture.json --title "Lecture Title" --date YYYY-MM-DD
```