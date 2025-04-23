#!/usr/bin/env python3
import argparse
import subprocess
import tempfile
from pathlib import Path
from groq import Groq

import os
from dotenv import load_dotenv
import sys
from jinja2 import Environment, FileSystemLoader
from datetime import datetime
import re

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and transcribe lectures into a timestamped markdown transcript"
    )
    parser.add_argument(
        "input",
        help="YouTube URL or local video file path"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output transcript LaTeX file (default: transcript.tex)",
        default="transcript.tex"
    )
    parser.add_argument(
        "-k", "--api-key",
        help="Groq API key (or set GROQ_API_KEY environment variable)",
        default=os.environ.get("GROQ_API_KEY")
    )
    parser.add_argument(
        "--cut", type=float, default=None,
        help="If set, only transcribe the first N minutes of the video/audio."
    )
    parser.add_argument(
        "--WhisperContext", type=str, default=None,
        help="String to use as initial_prompt for Whisper context biasing."
    )
    return parser.parse_args()

def extract_audio(source: str, wav_file: Path, cut_minutes: float = None):
    wav_file = Path(wav_file)
    if source.startswith(("http://", "https://")):
        cmd = [
            sys.executable, "-m", "yt_dlp", "-f", "bestaudio", "--extract-audio", "--audio-format", "wav",
            "-o", str(wav_file)
        ]
        if cut_minutes:
            # yt-dlp: use ffmpeg's -t via --postprocessor-args
            cmd += ["--postprocessor-args", f"-t {int(cut_minutes*60)}"]
        cmd.append(source)
    else:
        cmd = [
            "ffmpeg", "-y", "-i", source, "-vn",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1"
        ]
        if cut_minutes:
            cmd += ["-t", str(int(cut_minutes*60))]
        cmd.append(str(wav_file))
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # Split audio into 10-minute chunks
    chunk_duration = 600  # 10 minutes in seconds
    cmd = ['ffprobe', '-i', str(wav_file), '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=p=0']
    duration = float(subprocess.check_output(cmd).decode().strip())
    
    chunk_files = []
    chunks_dir = wav_file.parent / 'chunks'
    chunks_dir.mkdir(exist_ok=True)
    
    # Calculate number of full chunks and handle the last chunk specially
    num_full_chunks = int(duration) // chunk_duration
    remainder = duration - (num_full_chunks * chunk_duration)
    
    # Process full chunks
    for i in range(num_full_chunks):
        chunk_path = chunks_dir / f'chunk_{i:04d}.wav'
        cmd = [
            'ffmpeg', '-y', '-ss', str(i * chunk_duration), '-i', str(wav_file),
            '-t', str(chunk_duration), '-acodec', 'copy', str(chunk_path)
        ]
        subprocess.run(cmd, check=True)
        chunk_files.append(chunk_path)
    
    # Handle the last partial chunk if it exists and is long enough
    if remainder > 0.1:  # Ensure it's significantly longer than Groq's minimum requirement (0.01s)
        chunk_path = chunks_dir / f'chunk_{num_full_chunks:04d}.wav'
        cmd = [
            'ffmpeg', '-y', '-ss', str(num_full_chunks * chunk_duration), '-i', str(wav_file),
            '-acodec', 'copy', str(chunk_path)
        ]
        subprocess.run(cmd, check=True)
        chunk_files.append(chunk_path)
    
    return chunk_files


def transcribe(wav_files: list[Path], initial_prompt: str = None, api_key: str = None):
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Groq API key missing. Use --api-key or set GROQ_API_KEY environment variable")
    client = Groq(api_key=api_key)
    
    try:
        all_segments = []
        for i, chunk_path in enumerate(wav_files):
            print(f'Processing chunk {i+1}/{len(wav_files)}: {chunk_path.name}')
            
            # Verify audio duration before sending to API
            cmd = ['ffprobe', '-i', str(chunk_path), '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=p=0']
            chunk_duration = float(subprocess.check_output(cmd).decode().strip())
            
            if chunk_duration < 0.01:
                print(f"Skipping chunk {chunk_path.name} - duration too short ({chunk_duration:.3f}s)")
                continue
                
            with open(chunk_path, 'rb') as audio_file:
                response = client.audio.transcriptions.create(
                    file=audio_file,
                    model='whisper-large-v3',
                    response_format='verbose_json', 
                    timestamp_granularities=['segment']
                )
            # Apply timestamp offsets based on chunk position
            chunk_offset = i * 600
            # Process segments based on the actual response structure
            segments = response.segments if hasattr(response, 'segments') else response.get('segments', [])
            for seg in segments:
                # Handle both object attributes and dictionary keys
                if hasattr(seg, 'start') and hasattr(seg, 'end'):
                    seg.start += chunk_offset
                    seg.end += chunk_offset
                elif isinstance(seg, dict):
                    if 'start' in seg and 'end' in seg:
                        seg['start'] += chunk_offset
                        seg['end'] += chunk_offset
            all_segments.extend(segments)
        return {'segments': all_segments}
    except Exception as e:
        raise RuntimeError(f"Groq API error: {str(e)}")

def format_timestamp(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:06.3f}"

def format_timestamp(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f'{mins:02d}:{secs:02d}'

def write_latex(result: dict, output_path: Path):
    env = Environment(loader=FileSystemLoader('.'))
    env.filters['format_timestamp'] = format_timestamp
    template = env.get_template('transcript_template.tex.j2')
    latex_content = template.render(
        title="Lecture Transcript",
        date=datetime.now().strftime('%Y-%m-%d'),
        segments=result.get("segments", [])
    )
    
    # Post-process math expressions
    latex_content = re.sub(r'(\\$)(.*?)(\\$)', r'$\2$', latex_content)
    output_path.write_text(latex_content, encoding="utf-8")
    print(f"LaTeX transcript saved to {output_path}")


def main():
    load_dotenv()
    args = parse_args()
    try:
        inp = args.input
        out_md = Path(args.output)
        with tempfile.TemporaryDirectory() as tmp:
            wav_file = Path(tmp) / 'audio.wav'
            chunk_files = extract_audio(inp, wav_file, cut_minutes=args.cut)
            res = transcribe(chunk_files, initial_prompt=args.WhisperContext, api_key=args.api_key)
            write_latex(res, out_md)
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()