# LecToNotes

LecToNotes is a software package for extracting text resources from technical research mathematics lecture videos. It processes both the spoken audio and blackboard content, converting them into structured JSON data that can be viewed through a web-based dashboard.

## 📋 Prerequisites

Before getting started, make sure you have the following installed:

- **Python 3.8+**
- **Node.js 18+** (for the dashboard)
- **FFmpeg** (for audio processing)
- API keys for:
  - **Groq** (optional, for enhanced transcription)
  - **OpenRouter** (required for OCR processing)

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/DaveArcher18/LecToNotes.git
   cd LecToNotes
   ```

2. **Set up API keys**
   Create a `.env` file in the root directory with the following content:
   ```
   GROQ_API_KEY=your_groq_api_key_here
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
   - Extract transcript from the video
   - Extract board images
   - Process board images with OCR
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
- **whisper**: For local transcription (optional)
- **groq**: For cloud-based transcription (optional)
- **librosa, soundfile**: For audio preprocessing (optional)
- **yt-dlp**: For YouTube video downloading
- **tensorflow**: For enhanced board detection (optional)

### API Keys

The system uses two API services:

1. **OpenRouter API** (required for OCR):
   - Sign up at [OpenRouter](https://openrouter.ai/)
   - Get your API key from the dashboard
   - Add it to your `.env` file: `OPENROUTER_API_KEY=your_key_here`

2. **Groq API** (optional, for enhanced transcription, required for summarising transcript):
   - Sign up at [Groq](https://groq.com/)
   - Get your API key from the dashboard
   - Add it to your `.env` file: `GROQ_API_KEY=your_key_here`

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
# Extract transcript only
make extract_transcript VIDEO=/path/to/lecture.mp4 TITLE="Physics Lecture"

# Extract board images only
make extract_boards VIDEO=/path/to/lecture.mp4 TITLE="Chemistry Lecture"

# Process board images with OCR
make ocr_boards TITLE="Chemistry Lecture"

# Launch the dashboard
make launch_dashboard
```

### Processing YouTube Videos

You can directly process videos from YouTube:

```bash
make process_video VIDEO="https://youtube.com/watch?v=VIDEO_ID" TITLE="Calculus Lecture"
```

### Using Groq for Transcription

For enhanced transcription quality, you can use Groq's API:

```bash
make process_video VIDEO=/path/to/lecture.mp4 TITLE="Quantum Mechanics" USE_GROQ=true
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
  - Make sure you've created a `.env` file with the required API keys
  - Check that the API keys are correctly formatted with no spaces

- **Error: ImportError: No module named 'whisper'**
  - Run `make install_deps` to install all required dependencies
  - Alternatively, install whisper manually: `pip install -U openai-whisper`

- **Error: FFmpeg not found**
  - Install FFmpeg: 
    - macOS: `brew install ffmpeg`
    - Ubuntu: `sudo apt install ffmpeg`
    - Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html)

- **Dashboard shows empty content**
  - Check that files were correctly copied to `dashboard/public/OUTPUT/`
  - Verify there are no errors in the browser console

## 📄 License

[MIT](LICENSE)