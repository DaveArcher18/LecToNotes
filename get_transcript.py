import argparse
import base64
import json
import os
import re
import tempfile
import time
import hashlib
import sys  # Add missing sys import
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import ffmpeg  # type: ignore
import numpy as np
from dotenv import load_dotenv
from pydub import AudioSegment  # type: ignore
from scipy import signal
from tqdm import tqdm

try:
    import librosa
    import soundfile as sf
    librosa_available = True
except ImportError:
    librosa_available = False

try:
    import whisper  # type: ignore
    whisper_available = True
except ImportError:
    whisper_available = False

# ────────────────────────────────────────────────────────────────────────────
load_dotenv()

# OpenRouter configuration for DeepSeek Prover
import requests
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
SUMMARY_MODEL = "deepseek/deepseek-prover-v2:free"
OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "HTTP-Referer": "https://github.com/DaveArcher18/LecToNotes",
    "X-Title": "LecToNotes Transcript", 
    "Content-Type": "application/json"
}

RETRY_DELAY = 2  # seconds

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def check_environment():
    """Check if the environment is properly set up for transcription and summarization."""
    errors = []
    warnings = []
    
    # Check for API keys
    if not os.getenv("OPENROUTER_API_KEY"):
        errors.append("OPENROUTER_API_KEY is missing in the .env file. Required for DeepSeek Prover summarization.")
    
    # Check for required dependencies
    if not whisper_available:
        errors.append("Local Whisper not installed. Unable to perform transcription.")
    
    if not librosa_available:
        warnings.append("Librosa not installed. Audio preprocessing will be skipped.")
    
    # Print warnings and errors
    if warnings:
        print("\n[WARNINGS]")
        for warning in warnings:
            print(f"⚠️  {warning}")
    
    if errors:
        print("\n[ERRORS]")
        for error in errors:
            print(f"❌ {error}")
        print("\nPlease fix the above errors before running the script.")
        return False
    
    # No errors, environment is ready
    if not warnings:
        print("✅ Environment check passed. All dependencies and API keys are in place.")
    else:
        print("✅ Environment check passed with warnings.")
    
    return True

def hh_mm_ss(seconds: float) -> str:
    """Convert seconds to HH_MM_SS format."""
    seconds = int(seconds)
    h, m = divmod(seconds, 3600)
    m, s = divmod(m, 60)
    return f"{h:02d}_{m:02d}_{s:02d}"


def download_youtube(url: str, out_dir: Path) -> Path:
    """Download video from YouTube."""
    import yt_dlp  # late import
    opts = {
        "outtmpl": str(out_dir / "%(_id)s.%(ext)s"),
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
        "merge_output_format": "mp4",
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return out_dir / f"{info['id']}.mp4"


def extract_audio(video: Path, wav: Path):
    """Extract high-quality audio from video file."""
    (ffmpeg.input(str(video))
           .output(str(wav), ac=1, ar=48000, format="wav", loglevel="error")
           .overwrite_output()
           .run())


def preprocess_audio(wav_path: Path, out_path: Optional[Path] = None) -> Path:
    """Apply audio preprocessing to improve transcription quality."""
    if not librosa_available:
        print("[WARN] librosa not available, skipping audio preprocessing")
        return wav_path
    
    print(f"[INFO] Preprocessing audio file {wav_path}...")
    
    # Use the same path if output path is not specified
    if out_path is None:
        out_path = wav_path.with_name(f"{wav_path.stem}_processed.wav")
    
    # Load audio
    y, sr = librosa.load(wav_path, sr=None)
    
    # Apply preprocessing steps
    # 1. Noise reduction
    y_denoised = librosa.effects.preemphasis(y)
    
    # 2. Normalization
    y_normalized = librosa.util.normalize(y_denoised)
    
    # 3. Remove silence (not too aggressive)
    y_active = librosa.effects.trim(y_normalized, top_db=30)[0]
    
    # 4. Apply slight high-pass filter to improve speech clarity
    b, a = signal.butter(5, 100/(sr/2), 'highpass')
    y_filtered = signal.filtfilt(b, a, y_active)
    
    # 5. Final normalization
    y_final = librosa.util.normalize(y_filtered)
    
    # Save processed audio
    sf.write(out_path, y_final, sr)
    
    print(f"[INFO] Processed audio saved to {out_path}")
    return out_path


def chunk_wav(wav: Path, length_sec: int = 300, overlap_sec: int = 10) -> List[Path]:
    """Split audio into chunks with overlap."""
    audio = AudioSegment.from_wav(wav)
    paths = []
    
    for i, start in enumerate(range(0, len(audio), (length_sec - overlap_sec) * 1000)):
        seg = audio[start:start + length_sec * 1000]
        p = wav.with_name(f"chunk_{i:03d}.wav")
        seg.export(p, format="wav")
        paths.append(p)
    
    return paths


def read_context() -> str:
    """Read context from WhisperContext.txt file."""
    ctx = Path(__file__).with_name("WhisperContext.txt")
    if not ctx.exists():
        return ""
    # Read the entire context
    full_context = ctx.read_text(encoding="utf-8")
    max_length = 500  # Keep context under 500 characters
    if len(full_context) > max_length:
        print(f"[WARN] WhisperContext.txt is {len(full_context)} characters long. Truncating to {max_length} characters.")
        return full_context[:max_length] + "..."
    return full_context


def generate_context(transcript_so_far: str, segment_idx: int) -> str:
    """Generate context for transcription by combining file context and transcript history."""
    # Get static context
    context = read_context()
    
    # Truncate the static context if it's too long (leave room for transcript history)
    max_context_length = 400  # Reduced from original to leave room for transcript history
    if len(context) > max_context_length:
        context = context[:max_context_length] + "..."
    
    # If this isn't the first segment, include some previous transcript for context
    if segment_idx > 0 and transcript_so_far:
        # Extract the last ~300 words from transcript so far as context (reduced from 500)
        words = transcript_so_far.split()
        max_words = 300  # Reduced from 500 to keep context shorter
        previous_text = " ".join(words[-max_words:]) if len(words) > max_words else transcript_so_far
        
        # Add a short segment of the transcript
        context += f"\n\nPrevious transcript:\n{previous_text[-400:]}\n\nContinuation:"
    
    return context


def detect_and_fix_repetition(text: str) -> str:
    """Detect and fix repetitive phrases in transcript."""
    # Remove exact duplicate sentences
    lines = text.split('. ')
    seen = set()
    result = []
    
    for line in lines:
        line_clean = line.strip().lower()
        if line_clean and line_clean not in seen:
            seen.add(line_clean)
            result.append(line)
    
    # Fix repetition of phrases like "historically speaking, it's not very much a"
    pattern = re.compile(r'(.{10,50})\s+\1(\s+\1){2,}', re.DOTALL)
    cleaned_text = '. '.join(result)
    
    while True:
        match = pattern.search(cleaned_text)
        if not match:
            break
        
        # Replace the repetition with a single instance
        cleaned_text = cleaned_text[:match.start()] + match.group(1) + cleaned_text[match.end():]
    
    return cleaned_text


def load_or_create_transcript(out_path: Path, num_chunks: int) -> List[Dict[str, Any]]:
    """Load existing transcript or create a new one."""
    if out_path.exists():
        with open(out_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Create a new transcript file with placeholders
    return [{
        "start": hh_mm_ss(i * 300),
        "end": hh_mm_ss(i * 300 + 300),
        "content": "",
        "summary": ""
    } for i in range(num_chunks)]


def save_transcript(transcript: List[Dict[str, Any]], out_path: Path) -> None:
    """Save transcript to file with proper formatting."""
    # Create a backup if file exists
    if out_path.exists():
        backup_path = out_path.with_name(f"{out_path.stem}_backup_{int(time.time())}.json")
        out_path.rename(backup_path)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2)


# ---------------------------------------------------------------------------
# transcription engine (local Whisper only)
# ---------------------------------------------------------------------------
def transcribe_local(wav: Path, prompt: str, model_name: str = "medium") -> str:
    """Transcribe audio using local Whisper model."""
    if not whisper_available:
        raise RuntimeError("Local whisper not installed")
    model = whisper.load_model(model_name)
    out = model.transcribe(
        str(wav),
        initial_prompt="English academic lecture transcript: " + prompt,
        task="transcribe",
        temperature=0,
        best_of=5,
        beam_size=5,
        condition_on_previous_text=True
    )
    return out["text"].strip()


def generate_summary(content: str, previous_summaries: List[str] = None) -> str:
    """Generate a summary for a transcript segment using DeepSeek Prover via OpenRouter."""
    if not os.getenv("OPENROUTER_API_KEY"):
        print("[ERROR] OpenRouter API key not found in .env file. Please add OPENROUTER_API_KEY to your .env file.")
        return "Summary generation failed. OPENROUTER_API_KEY is missing."
    previous_context = ""
    if previous_summaries and len(previous_summaries) > 0:
        previous_context = "Previous segment summaries:\n" + "\n".join([
            f"Segment {i+1}: {summary}" for i, summary in enumerate(previous_summaries[-3:])
        ])
    prompt = f"""
You are an expert academic summarizer specializing in technical mathematics lectures.

Please summarize the following 5-minute transcript segment from a mathematics lecture. The summary should have exactly two paragraphs:

1. First paragraph: A comprehensive, information-rich summary capturing the key concepts, definitions, and mathematical relationships discussed.
2. Second paragraph: Start with "Added motivation:" and explain why these concepts matter in the broader context of the lecture if possible.

Use precise mathematical terminology and maintain the integrity of the technical content. The summary should be approximately 150 words total.

{previous_context}

TRANSCRIPT SEGMENT:
{content}
"""
    messages = [
        {"role": "system", "content": "You are an expert academic summarizer specializing in technical mathematics."},
        {"role": "user", "content": prompt}
    ]
    payload = {
        "model": SUMMARY_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1000,
    }
    max_retries = 3
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            response = requests.post(
                OPENROUTER_ENDPOINT, 
                headers=OPENROUTER_HEADERS,
                json=payload,
                timeout=60
            )
            if response.status_code != 200:
                print(f"[ERROR] OpenRouter API error (HTTP {response.status_code}): {response.text}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                break
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"[ERROR] Failed to generate summary: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"[ERROR] Failed to generate summary after {max_retries} attempts: {e}")
    return "Summary generation failed. Please try again later."


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def process_transcript(transcript_path: str, summarize_only: bool = False):
    """Process an existing transcript to add summaries."""
    transcript_path = Path(transcript_path)
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    
    # Skip entries that already have summaries
    entries_to_process = [entry for entry in transcript if not entry.get('summary')]
    if not entries_to_process:
        print("[INFO] All transcript entries already have summaries.")
        return
    
    print(f"[INFO] Generating summaries for {len(entries_to_process)} entries using DeepSeek Prover...")
    
    # Keep track of previous summaries for context
    previous_summaries = []
    for i, entry in enumerate(tqdm(transcript)):
        if entry.get('summary'):
            previous_summaries.append(entry['summary'])
            continue
            
        if not entry.get('content'):
            print(f"[WARN] Entry {i} has no content to summarize.")
            continue
            
        print(f"[INFO] Generating summary for segment {i+1}/{len(transcript)} with DeepSeek Prover...")
        entry['summary'] = generate_summary(entry['content'], previous_summaries)
        previous_summaries.append(entry['summary'])
        
        # Save after each summary to enable recovery
        save_transcript(transcript, transcript_path)
    
    print(f"[DONE] All summaries generated and saved to {transcript_path}")


def main():
    ap = argparse.ArgumentParser(description="Get transcript from YouTube or mp4 using Whisper.")
    ap.add_argument("input", nargs='?', help="YouTube URL or local mp4")
    ap.add_argument("--out", default="transcript.json", help="Output JSON file")
    ap.add_argument("--chunk-size", type=int, default=300, help="Chunk size in seconds (default: 300)")
    ap.add_argument("--overlap", type=int, default=10, help="Overlap between chunks in seconds (default: 10)")
    ap.add_argument("--skip-preprocessing", action="store_true", help="Skip audio preprocessing")
    ap.add_argument("--summarize-only", action="store_true", help="Only generate summaries for existing transcript using DeepSeek Prover")
    ap.add_argument("--skip-env-check", action="store_true", help="Skip environment validation check")
    ap.add_argument("--whisper-model", default="medium", choices=["tiny", "base", "small", "medium", "large"], help="Which Whisper model to use (default: medium)")
    args = ap.parse_args()

    # Check environment unless skipped
    if not args.skip_env_check and not check_environment():
        sys.exit(1)

    # For summarize-only mode, process existing transcript and exit
    if args.summarize_only:
        print(f"[INFO] Summarize-only mode: Processing existing transcript at {args.out}")
        # Check for OpenRouter API key
        if not os.getenv("OPENROUTER_API_KEY"):
            print("[ERROR] OpenRouter API key not found in .env file. Please add OPENROUTER_API_KEY to your .env file.")
            sys.exit(1)
        process_transcript(args.out, True)
        return

    # Validate input for transcription mode
    if not args.input:
        ap.error("the input argument is required unless using --summarize-only")

    if not whisper_available:
        print("[ERROR] Local Whisper model not available.")
        print("        Please install Whisper: pip install -U openai-whisper")
        sys.exit(1)

    print(f"[INFO] Processing video: {args.input}")
    print(f"[INFO] Output will be saved to: {args.out}")

    work = Path(tempfile.mkdtemp())
    print(f"[INFO] Created temporary working directory: {work}")

    # Acquire video
    if re.match(r"https?://", args.input):
        print(f"[INFO] Downloading YouTube video: {args.input}")
        try:
            video = download_youtube(args.input, work)
            print(f"[INFO] Downloaded video to: {video}")
        except Exception as e:
            print(f"[ERROR] Failed to download YouTube video: {e}")
            print("        Please check your internet connection and the video URL.")
            sys.exit(1)
    else:
        video = Path(args.input).resolve()
        print(f"[INFO] Using local video file: {video}")
        if not video.exists():
            print(f"[ERROR] Video file not found: {video}")
            print("        Please check that the file exists and the path is correct.")
            sys.exit(1)

    # Extract and preprocess audio
    try:
        print(f"[INFO] Extracting audio from video...")
        wav = work / "audio.wav"
        extract_audio(video, wav)
        print(f"[INFO] Audio extracted to: {wav}")

        if not args.skip_preprocessing and librosa_available:
            wav = preprocess_audio(wav)
        elif not librosa_available and not args.skip_preprocessing:
            print("[WARN] Librosa not available, skipping audio preprocessing.")
            print("       Install librosa for better audio quality: pip install librosa soundfile")
    except Exception as e:
        print(f"[ERROR] Failed to extract audio: {e}")
        print("        Make sure FFmpeg is installed and working properly.")
        sys.exit(1)

    # Split audio into chunks with overlap
    try:
        chunks = chunk_wav(wav, args.chunk_size, args.overlap)
        print(f"[INFO] Split audio into {len(chunks)} chunks of {args.chunk_size} seconds each (with {args.overlap}s overlap)")
    except Exception as e:
        print(f"[ERROR] Failed to split audio into chunks: {e}")
        sys.exit(1)

    # Read initial context
    base_prompt = read_context()
    if base_prompt:
        print(f"[INFO] Loaded context prompt from WhisperContext.txt ({len(base_prompt)} characters)")

    # Initialize output path and ensure its directory exists
    out_path = Path(args.out).resolve()
    os.makedirs(out_path.parent, exist_ok=True)
    print(f"[INFO] Output will be saved to: {out_path}")

    # Initialize or load transcript
    transcriptions = load_or_create_transcript(out_path, len(chunks))
    if out_path.exists():
        print(f"[INFO] Found existing transcript at {out_path}. Will resume from last position.")

    # Keep track of the full transcript for context
    full_transcript = ""
    previous_summaries = []

    # Count how many segments we need to process
    segments_to_process = [i for i, entry in enumerate(transcriptions) if not entry["content"]]
    if len(segments_to_process) < len(transcriptions):
        print(f"[INFO] Found {len(transcriptions) - len(segments_to_process)} already processed segments.")

    if not segments_to_process:
        print(f"[INFO] All segments already processed. Moving to summary generation if needed.")

    # Process each chunk
    for idx, chunk in enumerate(chunks):
        entry = transcriptions[idx]

        # Skip if already processed
        if entry["content"]:
            full_transcript += entry["content"] + " "
            if entry.get("summary"):
                previous_summaries.append(entry["summary"])
            continue

        print(f"[INFO] Transcribing segment {idx+1}/{len(chunks)} …")

        # Generate context for this chunk
        prompt = generate_context(full_transcript, idx)

        try:
            # Transcribe
            start_time = time.time()
            print(f"[INFO] Using local Whisper model ({args.whisper_model}) for segment {idx+1}...")
            text = transcribe_local(chunk, prompt, model_name=args.whisper_model)
            elapsed = time.time() - start_time
            print(f"[INFO] Transcription completed in {elapsed:.1f} seconds.")

            # Postprocess - fix repetitions
            text = detect_and_fix_repetition(text)

            # Add new content to the running transcript
            full_transcript += text + " "

            # Update entry
            entry["content"] = text

            # Generate summary
            print(f"[INFO] Generating summary for segment {idx+1}...")
            entry["summary"] = generate_summary(text, previous_summaries)
            previous_summaries.append(entry["summary"])

            # Save after each chunk to enable recovery
            save_transcript(transcriptions, out_path)
            print(f"[INFO] Saved progress to {out_path}")

        except Exception as e:
            print(f"[ERROR] Segment {idx} failed: {e}")
            save_transcript(transcriptions, out_path)
            print(f"[INFO] Partial transcript saved to {out_path}")
            print(f"[INFO] Run again with the same arguments to resume from segment {idx}")
            break  # keep partial JSON intact

    print(f"[DONE] Transcript saved to {out_path}")

    # Cleanup
    try:
        import shutil
        shutil.rmtree(work)
        print(f"[INFO] Cleaned up temporary files")
    except:
        print(f"[WARN] Failed to clean up temporary directory: {work}")


if __name__ == "__main__":
    main()
