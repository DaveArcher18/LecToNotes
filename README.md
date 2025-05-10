# LecToNotes

LecToNotes is a software package for extracting text resources from technical research mathematics lecture videos. It processes both the spoken audio and blackboard content, converting them into structured JSON data that can be viewed through a web-based dashboard.

## 📋 Prerequisites

Before getting started, make sure you have the following installed:

- **Python 3.8+**
- **Node.js 18+** (for the dashboard)
- **FFmpeg** (for audio processing)
- API key for:
  - **OpenRouter** (required for OCR processing and transcript summarization)

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/DaveArcher18/LecToNotes.git
   cd LecToNotes
   ```

2. **Set up API keys**
   Create a `.env` file in the root directory with the following content:
   ```
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

3. **Install dependencies**
   ```bash
   make install_deps
   ```
   This installs both Python and Node.js dependencies.

4. **Process a video**
   ```bash
   make process_video VIDEO=/path/to/lecture.mp4 TITLE="Linear Algebra Lecture 1"
   ```
   This command will:
   - Extract transcript from the video using the local Whisper model
   - Extract board images
   - Process board images with OCR (via OpenRouter)
   - Generate summaries for transcript segments (via OpenRouter)
   - Copy the results to the dashboard
   - Launch the dashboard web interface

5. **Access the dashboard**
   The dashboard will automatically launch at http://localhost:5173

## 🔧 Detailed Setup

### Python Dependencies

The project requires several Python packages. Use the following commands to install them:

```bash
# Option 1: Install using make
make install_deps

# Option 2: Install directly with pip
pip install -r requirements.txt
```

Key dependencies include:
- **numpy, opencv-python, scikit-image**: For image processing
- **ffmpeg-python, pydub**: For audio processing
- **openai-whisper**: For local transcription
- **librosa, soundfile**: For audio preprocessing (optional)
- **yt-dlp**: For YouTube video downloading
- **tensorflow**: For enhanced board detection (optional)
- **requests**: For making API calls to OpenRouter

### API Keys

The system uses the **OpenRouter API** (required for OCR and transcript summarization):
   - Sign up at [OpenRouter](https://openrouter.ai/)
   - Get your API key from the dashboard
   - Add it to your `.env` file: `OPENROUTER_API_KEY=your_key_here`
   - This key is used for both the OCR functionality on blackboard images and for generating summaries of the transcript segments.

### Dashboard Setup

The dashboard is automatically set up when you run `make process_video`. If you want to set it up manually:

```bash
# Install dependencies
cd dashboard
npm install

# Start the development server
npm run dev
```

## 📚 Using the Makefile

The project includes a comprehensive Makefile that automates the entire lecture processing pipeline.

### Main Commands

- **Process a complete video**: 
  ```bash
  make process_video VIDEO=/path/to/lecture.mp4 TITLE="Lecture Title"
  ```
  This uses the default medium Whisper model. To use a different model (e.g., large for higher accuracy):
  ```bash
  make process_video VIDEO=/path/to/lecture.mp4 TITLE="Lecture Title" WHISPER_MODEL=large
  ```

- **Install all dependencies**:
  ```bash
  make install_deps
  ```

- **Clean up all generated files**:
  ```bash
  make clean
  ```

- **Show help information**:
  ```bash
  make help
  ```

### Individual Pipeline Steps

You can also run individual steps of the pipeline:

```bash
# Extract transcript only (default medium model)
make extract_transcript VIDEO=/path/to/lecture.mp4 TITLE="Physics Lecture"

# Extract transcript only (large model)
make extract_transcript VIDEO=/path/to/lecture.mp4 TITLE="Physics Lecture" WHISPER_MODEL=large

# Extract board images only
make extract_boards VIDEO=/path/to/lecture.mp4 TITLE="Chemistry Lecture"

# Process board images with OCR
make ocr_boards TITLE="Chemistry Lecture"

# Launch the dashboard
make launch_dashboard
```

### Direct Script Usage

For testing or more control, you can also run the scripts directly:

#### Transcript Extraction (get_transcript.py)

Usage:

    python get_transcript.py <input> [--out OUTPUT] [--chunk-size N] [--overlap N] [--skip-preprocessing] [--summarize-only] [--skip-env-check] [--whisper-model MODEL]

Arguments:
  input                YouTube URL or local mp4 file
  --out                Output JSON file (default: transcript.json)
  --chunk-size         Chunk size in seconds (default: 300)
  --overlap            Overlap between chunks in seconds (default: 10)
  --skip-preprocessing Skip audio preprocessing
  --summarize-only     Only generate summaries for existing transcript using DeepSeek Prover (via OpenRouter)
  --skip-env-check     Skip environment validation check
  --whisper-model      Which Whisper model to use (tiny, base, small, medium, large). Default: medium. For best results, use 'medium' or 'large'.

Example:

    python get_transcript.py habiro_cohomology03.mp4 --out transcript.json --whisper-model large

#### Board Extraction (get_boards.py)

```bash
# Extract board images
python get_boards.py /path/to/lecture.mp4 -o output_dir

# Process extracted boards with OCR (requires transcript.json for context)
python LLM_OCR.py output_dir/boards.json --transcript transcript.json
```

### Processing YouTube Videos

You can directly process videos from YouTube:

```bash
make process_video VIDEO="https://youtube.com/watch?v=VIDEO_ID" TITLE="Calculus Lecture"
```

## 📊 Output Structure

All output files are stored in a directory named `<TITLE> OUTPUT`, with the following structure:

```
<TITLE> OUTPUT/
├── transcript.json         # Extracted transcript with summaries
└── boards/                 # Directory containing board images
    ├── board_*.jpg         # Extracted board images
    └── boards.json         # Metadata for board images with LaTeX transcriptions
```

The dashboard reads data from `dashboard/public/OUTPUT/`, which is automatically populated by `make process_video`.

## 📋 Output Format

### transcript.json

```json
[
  {
    "start": "00_00_00",
    "end": "00_05_00",
    "content": "Welcome to today's lecture on Habiro Cohomology...",
    "summary": "This segment introduces the concept of Habiro Cohomology..."
  },
  ...
]
```

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

## 🔍 Troubleshooting

- **Error: API key not found**
  - Make sure you've created a `.env` file with your `OPENROUTER_API_KEY`.
  - Check that the API key is correctly formatted with no spaces.

- **Error: ImportError: No module named 'whisper' or 'openai-whisper'**
  - Run `make install_deps` to install all required dependencies.
  - Alternatively, install whisper manually: `pip install -U openai-whisper`

- **Error: FFmpeg not found**
  - Install FFmpeg:
    - macOS: `brew install ffmpeg`
    - Ubuntu: `sudo apt install ffmpeg`
    - Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html)

- **Dashboard shows empty content**
  - Check that files were correctly copied to `dashboard/public/OUTPUT/`
  - Verify there are no errors in the browser console (usually F12 to open developer tools).

- **Low quality transcription or OCR**
  - For transcription, try using a larger Whisper model (e.g., `--whisper-model large` with `get_transcript.py` or `WHISPER_MODEL=large` with `make`). Note that larger models require more VRAM and processing time.
  - Ensure good audio quality in the input video.
  - For OCR, ensure blackboard images are clear and well-lit.

## 📄 License

[MIT](LICENSE)