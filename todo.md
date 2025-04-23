# Task checklist

This checklist tracks the progress of implementing the core functionalities.

- [ ] Find a way to populate `--WhisperContext` to improve transcription accuracy of technical terms.
- [x] Setup Project Structure & Basic Files (Implied: `.gitignore`, `project_specs.md`)
- [x] Implement Video Download/Access (`yt-dlp`)
- [x] Implement Audio Extraction (`ffmpeg`)
- [x] Implement Audio Transcription (Groq API - Whisper) - Initial setup likely in `get_transcript.py`
- [ ] Make a requirements.txt file and setup a virtual environment
- [ ] Implement Transcript Structuring (OpenRouter LLM API)
- [~] Implement Key Video Frame Sampling (OpenCV/PySceneDetect) - Initial script created (`frame_sampler.py`)
- [ ] Implement Blackboard Region Detection & Isolation (OpenCV)
- [ ] Implement Blackboard Region Enhancement (OpenCV)
- [ ] Implement Mathematical OCR (Surya/pix2tex)
- [ ] Implement Visual Content Structuring (OpenRouter LLM API)
- [ ] Implement Audio/Visual Content Alignment (Timestamps/Embeddings)
- [x] Create LaTeX Document Template (Jinja2) - `transcript_template.tex.j2` exists
- [ ] Implement Final LaTeX Document Generation (Jinja2 Integration)
- [ ] Integrate all components into a cohesive pipeline.
- [ ] Perform End-to-End Testing.
- [ ] Refine parameters, prompts, and error handling.
- [ ] Finalize Documentation (README).