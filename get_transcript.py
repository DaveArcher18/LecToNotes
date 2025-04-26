import argparse
import base64
import json
import os
import re
import tempfile
from pathlib import Path

import ffmpeg  # type: ignore
from dotenv import load_dotenv
from pydub import AudioSegment  # type: ignore
from groq import Groq  # type: ignore

try:
    import whisper  # type: ignore
except ImportError:
    whisper = None

# ────────────────────────────────────────────────────────────────────────────
load_dotenv()
client = Groq()  # uses GROQ_API_KEY from .env

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def hh_mm_ss(seconds: float) -> str:
    seconds = int(seconds)
    h, m = divmod(seconds, 3600)
    m, s = divmod(m, 60)
    return f"{h:02d}_{m:02d}_{s:02d}"


def download_youtube(url: str, out_dir: Path) -> Path:
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
    (ffmpeg.input(str(video))
           .output(str(wav), ac=1, ar=16000, format="wav", loglevel="error")
           .overwrite_output()
           .run())


def chunk_wav(wav: Path, length_sec: int = 300) -> list[Path]:
    audio = AudioSegment.from_wav(wav)
    paths = []
    for i, start in enumerate(range(0, len(audio), length_sec * 1000)):
        seg = audio[start:start + length_sec * 1000]
        p = wav.with_name(f"chunk_{i:03d}.wav")
        seg.export(p, format="wav")
        paths.append(p)
    return paths


def read_context() -> str:
    ctx = Path(__file__).with_name("WhisperContext.txt")
    return ctx.read_text(encoding="utf-8") if ctx.exists() else ""


# ---------------------------------------------------------------------------
# transcription engines
# ---------------------------------------------------------------------------

def transcribe_local(wav: Path, prompt: str) -> str:
    if whisper is None:
        raise RuntimeError("Local whisper not installed")
    model = whisper.load_model("medium")
    out = model.transcribe(
        str(wav),
        initial_prompt= "English transcript: " + prompt,
        task="transcribe",
        temperature=0
    )
    return out["text"].strip()

def transcribe_groq(wav: Path, prompt: str) -> str:
    with open(wav, "rb") as f:
        res = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=f,
            prompt="English transcript: " + prompt,
            temperature=0,
            suppress_tokens=[ 1, 2, 3, 4, 5 ] 
        )
    return res.text.strip()
# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Get transcript from YouTube or mp4 using Whisper.")
    ap.add_argument("input", help="YouTube URL or local mp4")
    ap.add_argument("--use-groq", action="store_true")
    ap.add_argument("--out", default="transcript.json")
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp())
    # Acquire video
    if re.match(r"https?://", args.input):
        video = download_youtube(args.input, work)
    else:
        video = Path(args.input).resolve()
        if not video.exists():
            raise FileNotFoundError(video)

    wav = work / "audio.wav"
    extract_audio(video, wav)

    chunks = chunk_wav(wav)
    prompt = read_context()

    out_path = Path(args.out).resolve()

    # -------------------------------- init / load JSON ----------------------
    if out_path.exists():
        transcriptions = json.loads(out_path.read_text())
    else:
        transcriptions = [{
            "start": hh_mm_ss(i * 300),
            "end": hh_mm_ss(i * 300 + 300),
            "content": ""
        } for i in range(len(chunks))]
        out_path.write_text(json.dumps(transcriptions, indent=2))

    # -------------------------------- iterate & fill ------------------------
    for idx, chunk in enumerate(chunks):
        entry = transcriptions[idx]
        if entry["content"]:
            continue  # already done
        print(f"[INFO] Transcribing segment {idx+1}/{len(chunks)} …")
        try:
            text = transcribe_groq(chunk, prompt) if args.use_groq else transcribe_local(chunk, prompt)
            entry["content"] = text
            out_path.write_text(json.dumps(transcriptions, indent=2))
        except Exception as e:
            print(f"[ERROR] Chunk {idx} failed: {e}")
            break  # keep partial JSON intact

    print(f"[DONE] Transcript saved to {out_path}")


if __name__ == "__main__":
    main()
