# Detailed Planning Document: Mathematics Lecture to LaTeX Conversion (API Focus)

**Version:** 1.1
**Date:** 2025-04-22

## 1. Project Goal & Scope

**Objective**: Develop a Python-based tool to convert mathematics lecture videos (YouTube/MP4) into a structured LaTeX (`.tex`) file. The tool will prioritize using free and open-source software (FOSS) or services with generous free tiers.

**Input**:
* YouTube URLs
* Local MP4 video files

**Output**:
* A single, compilable `.tex` file containing:
    * Timestamped transcript structured into paragraphs.
    * Mathematical expressions from audio/visuals formatted in LaTeX.
    * Blackboard content (text, equations) extracted via OCR and formatted in LaTeX.
    * Diagrams included as images (`.png` or `.jpg`) with generated captions.

**Core Functionality (API Focus)**:
* Download/Access Video.
* Extract Audio.
* Transcribe Audio using the **Groq API (Whisper model)**.
* Structure Transcript using a **LLM API (OpenRouter)**.
* Sample Key Video Frames using **OpenCV/PySceneDetect**.
* Detect, Isolate, and Enhance Blackboard Region using **OpenCV**.
* Perform Mathematical OCR using **local FOSS tools (Surya, pix2tex)**.
* Structure Visual Content using a **LLM API (OpenRouter)**.
* Align Audio and Visual Content using **timestamps and local sentence embeddings**.
* Generate Final LaTeX Document using **Jinja2**.

## 2. Core Technologies (FOSS/Freemium Focus)

| Category              | Recommended Tool(s)                                     | Cost Focus        | Notes                                                                 |
| :-------------------- | :------------------------------------------------------ | :---------------- | :-------------------------------------------------------------------- |
| Video Download        | `yt-dlp`                                                | FOSS              | Reliable, well-maintained.                                            |
| Audio Extraction      | `ffmpeg-python` (or `subprocess` + `ffmpeg`)            | FOSS              | Requires `ffmpeg` installation.                                       |
| Transcription         | Groq API (`whisper-large-v3` or `turbo`)                | Freemium          | Very fast, provides word timestamps. Free tier limits apply.          |
| Transcript Structuring| OpenRouter API                                          | Freemium          | Access various models via unified API; select low-cost/free options.  |
| Frame Sampling        | `PySceneDetect`                                         | FOSS              | Robust scene change detection with timestamps.                        |
|                       | *Alternative: `OpenCV` (manual differencing)* | *FOSS* | *Simpler but potentially less robust.* |
| Blackboard Processing | `OpenCV`                                                | FOSS              | Detection, perspective correction, enhancement.                       |
| Mathematical OCR      | `Surya`, `pix2tex`                                      | FOSS              | Evaluate both for accuracy on sample data. Requires local compute.    |
|                       | *Alternative: Local Multimodal LLM (`LLaVA`)* | *FOSS* | *Potential for diagrams but higher hallucination risk for math.* |
| Visual Structuring    | OpenRouter API (as above)                               | Freemium          | For basic structuring, captioning aid.                                |
|                       | `transformers` (for local image captioning)             | FOSS              | For generating diagram captions locally.                              |
| Alignment             | `sentence-transformers`                                 | FOSS              | Local semantic similarity calculation.                                |
| LaTeX Generation      | `Jinja2`                                                | FOSS              | Flexible templating engine.                                           |
| Compilation Check     | `subprocess` + `pdflatex`                               | FOSS              | Requires local LaTeX distribution (TeX Live, MiKTeX).                 |

## 3. Detailed Implementation Steps

### Step 0: Environment Setup

* **Action**: Create a Python virtual environment (e.g., using `venv` or `conda`).
* **Action**: Install core dependencies: `yt-dlp`, `ffmpeg-python`, `opencv-python`, `numpy`, `pydub` (for audio chunking), `requests`, `python-dotenv`, `openai` (for OpenRouter), `groq` (for Groq API), `jinja2`, `sentence-transformers`, `torch`, `transformers`.
* **Action**: Install `ffmpeg` system-wide (refer to OS-specific instructions).
* **Action**: Install a local LaTeX distribution (e.g., TeX Live, MiKTeX) if compilation checks are desired.
* **Action**: Set up API keys (Groq, OpenRouter) securely using environment variables (`.env` file and `python-dotenv`).
* **Action (Local AI Setup - Optional for Math OCR/Captioning)**:
    * Install `Surya` and/or `pix2tex`. Download required models.
    * Configure Hugging Face `transformers` for local image captioning. Download model weights.
    * *Note: This often requires specific C++ compilers, CUDA toolkits (for GPU), etc. Follow individual tool instructions carefully.*

### Step 1: Video Acquisition (Input: URL or Path)

* **Function**: `acquire_video(source)`
* **Input**: `source` (string: YouTube URL or local file path)
* **Logic**:
    1.  Check if `source` starts with `http`.
    2.  If yes (URL):
        * Use `yt-dlp` (via `subprocess` or Python API) to download the video.
        * Specify format (e.g., `bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best`).
        * Define a standard output path (e.g., `downloaded_video.mp4`).
        * Handle potential download errors (exceptions, non-zero return codes).
        * Return the path to the downloaded MP4 file.
    3.  If no (local path):
        * Check if the file exists and has a compatible extension (e.g., `.mp4`).
        * Return the validated local path.
* **Output**: Path to the local MP4 video file.

### Step 2: Audio Extraction (Input: Video Path)

* **Function**: `extract_audio(video_path)`
* **Input**: `video_path` (string)
* **Logic**:
    1.  Define the output audio path (e.g., `extracted_audio.wav`).
    2.  Use `ffmpeg-python` (or `subprocess` calling `ffmpeg`):
        * Input: `video_path`.
        * Options: `-vn` (no video), `-acodec pcm_s16le` (WAV format), `-ar 16000` (16kHz sample rate), `-ac 1` (mono).
        * Output: `extracted_audio.wav`.
    3.  Run the command, handling potential errors.
* **Output**: Path to the extracted WAV audio file.

### Step 3: Audio Transcription (Input: Audio Path)

* **Function**: `transcribe_audio_groq(audio_path)`
* **Input**: `audio_path` (string)
* **Logic (Groq API)**:
    1.  Initialize the Groq client using the API key from environment variables.
    2.  Check audio file size. If it exceeds Groq API limits (e.g., 100MB free tier), chunk the audio using `pydub`.
        * Split the audio into manageable chunks (e.g., 10-15 minutes or based on size). Consider slight overlaps between chunks.
    3.  For each audio chunk (or the full file if small enough):
        * Open the audio file/chunk in binary read mode.
        * Call `client.audio.transcriptions.create()`:
            * Pass the file object.
            * Specify the model (`whisper-large-v3` or `whisper-large-v3-turbo`).
            * Specify the language (e.g., `language="en"`).
            * Request `response_format="verbose_json"` to get detailed timestamps.
            * Check Groq documentation for the specific parameter to enable **word-level timestamps**.
        * Handle potential API errors (rate limits, etc.) with retries if necessary.
    4.  Combine the transcriptions from chunks:
        * Extract segment-level and word-level timestamps from the `verbose_json` responses.
        * Carefully stitch the transcript segments together, adjusting timestamps based on chunk start times and handling overlaps to avoid duplication or truncation (e.g., using LCS or alignment logic).
* **Output**: Structured transcript data (e.g., list of dictionaries: `[{'text': '...', 'start': 0.0, 'end': 5.2, 'words': [{'word': 'The', 'start': 0.1, 'end': 0.3}, ...]}, ...]`).

### Step 4: Transcript Structuring (Input: Raw Transcript Data)

* **Function**: `structure_transcript(transcript_data)`
* **Input**: `transcript_data` (structured data from Step 3)
* **Logic (OpenRouter API)**:
    1.  Combine segments of the raw transcript into larger chunks suitable for the LLM context window (respecting timestamps).
    2.  For each chunk:
        * Prepare a prompt for a selected OpenRouter model (e.g., `mistralai/mistral-7b-instruct:free`, `nousresearch/nous-hermes-2-mixtral-8x7b-dpo:free`).
        * Prompt Instructions: "Structure the following lecture transcript segment into paragraphs. Improve readability. Format any simple inline math using LaTeX delimiters ($...$). IMPORTANT: Preserve the original meaning and associate the original start/end timestamps with the corresponding structured paragraphs."
        * Use the `openai` library configured for OpenRouter to send the request.
        * Parse the response, extracting structured text and associated timestamps.
    3.  Combine results from chunks.
* **Output**: List of structured text blocks with associated timestamps (e.g., `[{'text': 'Paragraph 1...', 'start': 0.0, 'end': 15.5}, {'text': 'Paragraph 2...', 'start': 15.6, 'end': 30.1}]`).

### Step 5: Key Frame Sampling (Input: Video Path)

* **Function**: `sample_key_frames(video_path)`
* **Input**: `video_path` (string)
* **Logic (PySceneDetect)**:
    1.  Initialize `VideoManager` with `video_path`.
    2.  Add a `ContentDetector` (or `AdaptiveDetector`) to the `SceneManager`. Adjust threshold based on testing.
    3.  Call `detect_scenes()` on the `SceneManager`.
    4.  Retrieve the list of scene cuts (start/end timestamps).
    5.  For each scene, select representative key frames (e.g., the first frame after the cut, or frames at regular intervals within the scene). Store frame image and precise timestamp (using `FrameTimecode`).
* **Logic (Alternative: OpenCV)**:
    1.  Open video using `cv2.VideoCapture`.
    2.  Read frames at a fixed interval (e.g., every 1 second).
    3.  Convert current and previous sampled frames to grayscale.
    4.  Calculate difference (e.g., `cv2.absdiff`, then sum/count non-zero pixels) or SSIM (`skimage.metrics.structural_similarity`).
    5.  If the difference/similarity exceeds a threshold, store the current frame and its timestamp (`cap.get(cv2.CAP_PROP_POS_MSEC)`).
* **Output**: List of key frames with timestamps (e.g., `[(timestamp1, frame1_numpy_array), (timestamp2, frame2_numpy_array), ...]`).

### Step 6: Blackboard Processing (Input: Key Frames)

* **Function**: `process_blackboard_frame(frame)`
* **Input**: `frame` (NumPy array)
* **Logic (OpenCV)**:
    1.  **Detection**:
        * Convert frame to grayscale.
        * Apply Canny edge detection.
        * Find contours (`cv2.findContours`).
        * Approximate contours to polygons (`cv2.approxPolyDP`).
        * Filter for large quadrilaterals (potential blackboard). Heuristics based on area, aspect ratio.
        * *Fallback*: If detection fails, use a predefined region or the whole frame (with margins).
        * Get the 4 corner points of the detected region.
    2.  **Perspective Correction**:
        * Define target rectangle dimensions.
        * Calculate perspective transform matrix (`cv2.getPerspectiveTransform`).
        * Apply warp (`cv2.warpPerspective`) to get a flattened view of the blackboard.
    3.  **Enhancement**:
        * Apply CLAHE (`cv2.createCLAHE`) for adaptive contrast.
        * Apply noise reduction (`cv2.medianBlur` or `cv2.fastNlMeansDenoising`).
        * Apply adaptive thresholding (`cv2.adaptiveThreshold`) or Otsu's binarization (`cv2.threshold` + `cv2.THRESH_OTSU`) to get a binary image optimized for OCR. Experimentation is key here.
* **Output**: Processed (flattened, enhanced, binary) blackboard image (NumPy array).

### Step 7: Mathematical OCR (Input: Processed Blackboard Images with Timestamps)

* **Function**: `perform_math_ocr(processed_images_with_timestamps)`
* **Input**: List of tuples `[(timestamp, processed_image)]`
* **Logic (Surya)**:
    1.  Initialize Surya OCR components (detector, recognizer). Specify language (`en`) and target device (`cuda` or `cpu`).
    2.  For each `processed_image`:
        * Run Surya OCR (`surya.ocr.run_ocr`). This should handle layout analysis and provide text/LaTeX output with bounding boxes.
        * Extract the relevant LaTeX/text content.
        * Store the result associated with the `timestamp`.
* **Logic (Alternative: pix2tex)**:
    1.  Load the pix2tex model.
    2.  For each `processed_image`:
        * Convert NumPy array to PIL Image.
        * Call the pix2tex model prediction function.
        * Extract the generated LaTeX string.
        * Store the result associated with the `timestamp`.
* **Output**: List of OCR results with timestamps (e.g., `[(timestamp1, latex_string1), (timestamp2, latex_string2), ...]`). Handle potential errors/empty results.

### Step 8: Visual Content Structuring & Diagram Handling (Input: OCR Results)

* **Function**: `structure_visual_content(ocr_results)`
* **Input**: List `[(timestamp, ocr_content)]`
* **Logic**:
    1.  Initialize a local image captioning model (e.g., `Salesforce/blip-image-captioning-large` via `transformers`).
    2.  Iterate through `ocr_results`:
        * If `ocr_content` is primarily LaTeX/text:
            * Optionally, use an LLM (OpenRouter API) for basic validation/structuring (e.g., grouping multi-line equations). Prompt carefully: "Validate and structure this LaTeX snippet: [snippet]. Correct only obvious syntax errors."
            * Store the (potentially refined) LaTeX snippet with its timestamp.
        * If `ocr_content` indicates a diagram (e.g., very little text from OCR, or basic shape detection via OpenCV contours on the *processed* image):
            * Save the corresponding *processed* blackboard image temporarily.
            * Generate a caption using the local captioning model.
            * Optionally, refine the caption using an LLM (OpenRouter API) for mathematical context: "Refine this caption for a math lecture diagram: [caption]".
            * Store an instruction to include the image file and the final caption, associated with the timestamp.
* **Output**: List of structured visual elements (LaTeX snippets, image inclusion instructions with captions) with timestamps. (e.g., `[{'type': 'latex', 'content': '\\frac{d}{dx}x^2=2x', 'timestamp': 30.5}, {'type': 'image', 'path': 'frame_45.png', 'caption': 'Graph of y=x^2', 'timestamp': 45.2}]`).

### Step 9: Integration and Alignment (Input: Structured Transcript, Structured Visuals)

* **Function**: `integrate_content(structured_transcript, structured_visuals)`
* **Input**: Outputs from Step 4 and Step 8.
* **Logic**:
    1.  **Timestamp Alignment (Primary)**:
        * Create a timeline of events (transcript segments start/end, visual elements appear).
        * Iterate through transcript segments. For each segment, find visual elements whose timestamps fall within or near the segment's time range. Associate them.
    2.  **Semantic Refinement (Optional but Recommended)**:
        * Load a `sentence-transformers` model (e.g., `all-MiniLM-L6-v2`).
        * Generate embeddings for transcript segment text.
        * Generate embeddings for visual element text (extracted from LaTeX or captions).
        * Calculate cosine similarity between transcript segments and nearby (in time) visual elements.
        * Use high similarity scores (> threshold) to strengthen or adjust timestamp-based associations.
    3.  **Narrative Construction**:
        * Create a final ordered list representing the document flow.
        * Iterate through the transcript segments chronologically.
        * For each segment, add its text.
        * Add the associated visual elements (LaTeX or image instructions) immediately after the relevant text.
* **Output**: A single, ordered list representing the final document content flow (e.g., `[{'type': 'text', 'content': '...'}, {'type': 'latex', 'content': '...'}, {'type': 'image', 'path': '...', 'caption': '...'}, ...]`).

### Step 10: LaTeX Document Generation (Input: Integrated Content Stream)

* **Function**: `generate_latex(integrated_content, template_path='template.tex.j2')`
* **Input**: Output from Step 9, path to Jinja2 template.
* **Logic**:
    1.  Create a Jinja2 template (`template.tex.j2`) with:
        * Standard LaTeX preamble (`\documentclass`, `\usepackage{amsmath, amssymb, graphicx, geometry, utf8, T1}`, etc.).
        * `\title`, `\author`, `\date`, `\maketitle`, `\tableofcontents`.
        * `\begin{document}`.
        * A Jinja loop (`{% for item in content_stream %}`) iterating through the integrated content list.
        * Inside the loop, use Jinja conditionals (`{% if item.type == 'text' %}`):
            * If 'text', insert `{{ item.content }}`.
            * If 'latex', insert `{{ item.content }}` (ensure it's marked safe if needed, or handle escaping appropriately). Use display math (`\[ ... \]`) or inline math (`$ ... $`) based on context if possible.
            * If 'image', insert `\begin{figure}[htbp]\centering\includegraphics[width=0.8\textwidth]{{{ item.path }}}\caption{{{ item.caption }}}\label{fig:{{ loop.index }}}\end{figure}`.
        * `\end{document}`.
    2.  Load the Jinja2 environment and template.
    3.  Render the template, passing the `integrated_content` list as `content_stream`.
    4.  Write the rendered string to the output `.tex` file.
* **Output**: Final `.tex` file path.

### Step 11: Compilation Check (Optional) (Input: LaTeX File Path)

* **Function**: `check_latex_compilation(tex_path)`
* **Input**: Path to the generated `.tex` file.
* **Logic**:
    1.  Use `subprocess.run` to execute `pdflatex -interaction=nonstopmode -halt-on-error <tex_path>`.
    2.  Capture `stdout`, `stderr`, and the return code.
    3.  Check the return code. 0 usually indicates success.
    4.  Log success or failure, potentially including error messages from `stderr`.
* **Output**: Boolean indicating compilation success/failure.

## 4. Key Challenges & Mitigations (API Focus)

* **API Rate Limits & Costs**: Exceeding free tiers on Groq or OpenRouter can lead to errors or costs.
    * **Mitigation**: Implement robust audio chunking for Groq. Use designated free models on OpenRouter. Implement request batching and caching where appropriate. Clearly inform users about potential costs and limits. Monitor usage.
* **FOSS Math OCR Accuracy**: `Surya` and `pix2tex` might not match commercial APIs like Mathpix, especially for messy handwriting.
    * **Mitigation**: Extensive image preprocessing (Step 6) is crucial. Evaluate both `Surya` and `pix2tex` on diverse samples. Consider allowing manual correction or fallback to image inclusion for unrecognised complex equations.
* **Alignment Complexity**: Simple timestamp matching can fail. Semantic matching adds complexity.
    * **Mitigation**: Use precise word-level timestamps from Groq. Implement semantic refinement carefully. Allow configurable time windows for matching.
* **Setup Complexity (Local OCR/Captioning)**: Installing local AI tools (`Surya`, `transformers`) can be challenging.
    * **Mitigation**: Provide detailed setup instructions, potentially Dockerfiles. Clearly separate core functionality from optional local AI components.
* **Network Dependency**: Reliance on external APIs (Groq, OpenRouter) makes the tool dependent on network connectivity and API availability.
    * **Mitigation**: Implement robust error handling for network issues and API downtime. Use retries with backoff. Inform the user about failures.

## 5. Dependencies Summary

* **Python Packages**: `yt-dlp`, `ffmpeg-python`, `opencv-python`, `numpy`, `pydub`, `requests`, `python-dotenv`, `openai` (for OpenRouter), `groq`, `jinja2`, `sentence-transformers`, `torch`, `transformers` (for captioning), `PySceneDetect`, `scikit-image` (for SSIM if not using PySceneDetect), `surya-ocr` (and its dependencies), `pix2tex` (and its dependencies), `librosa`/`soundfile` (for audio loading).
* **System Tools**: `ffmpeg`, LaTeX Distribution (`pdflatex`), potentially C++ compiler, CUDA toolkit (for GPU if using local OCR/captioning).
* **AI Models (Local - Optional)**: Surya/pix2tex models, sentence-transformer models, image captioning models.

## 6. Next Steps

1.  **Setup & Environment Validation**: Ensure all core dependencies install and run correctly. Test `ffmpeg` and `pdflatex` commands. Test Groq and OpenRouter API connections.
2.  **Sample Data**: Gather diverse math lecture videos (different styles, topics, quality).
3.  **Component Implementation & Testing**: Build and test each function (Steps 1-11) individually using sample data. Pay special attention to:
    * Groq API integration, chunking logic, and word-level timestamp accuracy.
    * OpenRouter free model performance for structuring.
    * Blackboard enhancement effectiveness.
    * Surya vs. pix2tex OCR accuracy comparison (if using local OCR).
    * Alignment logic robustness.
4.  **Pipeline Integration**: Connect the components into the full pipeline.
5.  **End-to-End Testing**: Run the full pipeline on various sample videos. Monitor API usage.
6.  **Refinement**: Tune parameters (detection thresholds, OCR settings, alignment windows), improve error handling, and refine prompts based on test results.
7.  **Documentation**: Finalize README with setup, usage, API key management, and troubleshooting instructions.
