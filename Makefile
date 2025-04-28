# LecToNotes Makefile

# Default values
VIDEO ?= 
TITLE ?= "Untitled Lecture"
DATE ?= 
INTERVAL ?= 300
USE_GROQ ?= false

# Directories
OUTPUT_DIR = "$(TITLE) OUTPUT"
BOARDS_DIR = $(OUTPUT_DIR)/boards
MERGE_DIR = $(OUTPUT_DIR)/merge

# Output files
BOARDS_JSON = $(BOARDS_DIR)/boards.json
TRANSCRIPT_JSON = $(OUTPUT_DIR)/transcript.json
LECTURE_JSON = $(MERGE_DIR)/lecture.json

# Main target - process a video completely
# Note: extract_transcript is run before extract_boards to handle YouTube URL downloads
process: extract_transcript extract_boards ocr_boards merge

# Extract board images from video
extract_boards:
	@echo "Extracting board images from video..."
	@mkdir -p $(BOARDS_DIR)
	python get_boards.py -i $(VIDEO) -o $(BOARDS_DIR)

# Process board images with OCR
ocr_boards:
	@echo "Processing board images with OCR..."
	python LLM_OCR.py $(BOARDS_JSON)

# Extract transcript from video
extract_transcript:
	@echo "Extracting transcript from video..."
	@mkdir -p $(OUTPUT_DIR)
	if [ "$(USE_GROQ)" = "true" ]; then \
		python get_transcript.py $(VIDEO) --use-groq --out $(TRANSCRIPT_JSON); \
	else \
		python get_transcript.py $(VIDEO) --out $(TRANSCRIPT_JSON); \
	fi

# Merge boards and transcript
merge:
	@echo "Merging boards and transcript..."
	@mkdir -p $(MERGE_DIR)
	python merge/unified_merge.py -b $(BOARDS_JSON) -t $(TRANSCRIPT_JSON) -o $(LECTURE_JSON) \
		--title $(TITLE) $(if $(DATE),--date $(DATE),) --interval $(INTERVAL)

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf $(OUTPUT_DIR)

# Help target
help:
	@echo "LecToNotes Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make process VIDEO=path/to/video.mp4 TITLE=\"Lecture Title\" DATE=YYYY-MM-DD"
	@echo ""
	@echo "Targets:"
	@echo "  process            Run the complete pipeline"
	@echo "  extract_boards     Extract board images from video"
	@echo "  ocr_boards         Process board images with OCR"
	@echo "  extract_transcript Extract transcript from video"
	@echo "  merge              Merge boards and transcript"
	@echo "  clean              Remove generated files"
	@echo "  help               Show this help message"
	@echo ""
	@echo "Options:"
	@echo "  VIDEO              Path to input video file"
	@echo "  TITLE              Lecture title (default: \"Untitled Lecture\")"
	@echo "  DATE               Lecture date in YYYY-MM-DD format (optional)"
	@echo "  INTERVAL           Time interval for segmentation in seconds (default: 300)"
	@echo "  USE_GROQ           Use Groq API for transcription (true/false, default: false)"
	@echo ""
	@echo "Output:"
	@echo "  All output files will be stored in a directory named \"<TITLE> OUTPUT\""

.PHONY: process extract_boards ocr_boards extract_transcript merge clean help