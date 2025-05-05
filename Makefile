# --------------------------------------------------------------------
# LecToNotes Makefile
# --------------------------------------------------------------------
# A comprehensive build system for extracting content from math lecture videos
# 
# Targets:
#   make process_video VIDEO=path.mp4 TITLE="Title"   →  Process entire video pipeline
#   make extract_transcript VIDEO=path.mp4 TITLE="T"  →  Extract transcript only
#   make extract_boards VIDEO=path.mp4 TITLE="Title"  →  Extract board images only
#   make ocr_boards TITLE="Title"                     →  Process boards with OCR
#   make dashboard_copy TITLE="Title"                 →  Copy results to dashboard
#   make launch_dashboard                             →  Start the dashboard server
#   make clean TITLE="Title"                          →  Remove generated files
#   make install_deps                                 →  Install all dependencies
#   make help                                         →  Show detailed documentation
# --------------------------------------------------------------------

# Default values
VIDEO ?= 
TITLE ?= Untitled Lecture
INTERVAL ?= 300
USE_GROQ ?= false
SKIP_HUMANS ?= false
SKIP_OCR ?= false
HUMAN_CONFIDENCE ?= 0.4

# Directories
OUTPUT_DIR = "$(TITLE) OUTPUT"
BOARDS_DIR = "$(OUTPUT_DIR)/boards"
DASHBOARD_OUTPUT = dashboard/public/OUTPUT

# Output files
BOARDS_JSON = "$(BOARDS_DIR)/boards.json"
TRANSCRIPT_JSON = "$(OUTPUT_DIR)/transcript.json"

# Colors for prettier output
BOLD = \033[1m
GREEN = \033[32m
BLUE = \033[34m
YELLOW = \033[33m
RESET = \033[0m

# -------- Main targets -------------------------------------------------------

# Process a complete video through the entire pipeline
process_video:
	@if [ -z "$(VIDEO)" ]; then \
		echo "$(BOLD)$(YELLOW)Error: VIDEO parameter is required$(RESET)"; \
		echo "Usage: make process_video VIDEO=path/to/video.mp4 TITLE=\"Lecture Title\""; \
		exit 1; \
	fi
	@echo "$(BOLD)$(GREEN)Starting complete processing pipeline for: $(VIDEO)$(RESET)"
	@echo "$(BOLD)$(GREEN)Title: $(TITLE)$(RESET)"
	@$(MAKE) extract_transcript VIDEO="$(VIDEO)" TITLE="$(TITLE)" USE_GROQ="$(USE_GROQ)"
	@$(MAKE) extract_boards VIDEO="$(VIDEO)" TITLE="$(TITLE)" SKIP_HUMANS="$(SKIP_HUMANS)" HUMAN_CONFIDENCE="$(HUMAN_CONFIDENCE)"
	@$(MAKE) ocr_boards TITLE="$(TITLE)" SKIP_OCR="$(SKIP_OCR)"
	@$(MAKE) dashboard_copy TITLE="$(TITLE)"
	@$(MAKE) launch_dashboard
	@echo "$(BOLD)$(GREEN)Processing complete!$(RESET)"

# -------- Pipeline steps -----------------------------------------------------

# Extract board images from video
extract_boards:
	@if [ -z "$(VIDEO)" ]; then \
		echo "$(BOLD)$(YELLOW)Error: VIDEO parameter is required$(RESET)"; \
		echo "Usage: make extract_boards VIDEO=path/to/video.mp4 TITLE=\"Lecture Title\""; \
		exit 1; \
	fi
	@echo "$(BOLD)$(BLUE)Extracting board images from video: $(VIDEO)$(RESET)"
	@mkdir -p $(BOARDS_DIR)
	@echo "$(BOLD)$(BLUE)Using human detection to filter out lecturer images.$(RESET)"
	@echo "$(BOLD)$(BLUE)Use SKIP_HUMANS=true to disable this feature if needed.$(RESET)"
	@if [ "$(SKIP_HUMANS)" = "true" ]; then \
		python get_boards.py -i "$(VIDEO)" -o $(BOARDS_DIR) --preoptimize-ocr --enhance --skip-human-detection; \
	else \
		python get_boards.py -i "$(VIDEO)" -o $(BOARDS_DIR) --preoptimize-ocr --enhance --human-confidence $(HUMAN_CONFIDENCE); \
	fi || { \
		echo "$(BOLD)$(YELLOW)Error extracting board images. Check error messages above.$(RESET)"; \
		echo "$(BOLD)$(YELLOW)You can try adjusting parameters or use SKIP_HUMANS=true.$(RESET)"; \
		exit 1; \
	}

# Process board images with OCR
ocr_boards:
	@if [ ! -d $(OUTPUT_DIR) ]; then \
		echo "$(BOLD)$(YELLOW)Warning: Output directory not found. Creating it.$(RESET)"; \
		mkdir -p $(OUTPUT_DIR); \
	fi
	@if [ ! -f $(BOARDS_JSON) ]; then \
		echo "$(BOLD)$(YELLOW)Error: boards.json not found. Run extract_boards first.$(RESET)"; \
		exit 1; \
	fi
	@if [ ! -f $(TRANSCRIPT_JSON) ]; then \
		echo "$(BOLD)$(YELLOW)Warning: transcript.json not found. OCR will run without transcript context.$(RESET)"; \
		if [ "$(SKIP_OCR)" = "true" ]; then \
			echo "$(BOLD)$(YELLOW)Skipping OCR processing due to SKIP_OCR=true$(RESET)"; \
		else \
			echo "$(BOLD)$(BLUE)Processing board images with OCR (without transcript).$(RESET)"; \
			python LLM_OCR.py $(BOARDS_JSON) || { \
				echo "$(BOLD)$(YELLOW)OCR processing failed. Check OpenRouter API key in .env file.$(RESET)"; \
				echo "$(BOLD)$(YELLOW)Continuing pipeline without OCR. Board images will still be displayed.$(RESET)"; \
				echo "$(BOLD)$(YELLOW)You can try again later with: make ocr_boards TITLE=\"$(TITLE)\"$(RESET)"; \
			}; \
		fi; \
	else \
		echo "$(BOLD)$(BLUE)Processing board images with OCR for: $(TITLE)$(RESET)"; \
		echo "$(BOLD)$(BLUE)Using transcript at: $(TRANSCRIPT_JSON)$(RESET)"; \
		if [ "$(SKIP_OCR)" = "true" ]; then \
			echo "$(BOLD)$(YELLOW)Skipping OCR processing due to SKIP_OCR=true$(RESET)"; \
		else \
			python LLM_OCR.py $(BOARDS_JSON) --transcript $(TRANSCRIPT_JSON) || { \
				echo "$(BOLD)$(YELLOW)OCR processing failed. Check OpenRouter API key in .env file.$(RESET)"; \
				echo "$(BOLD)$(YELLOW)Continuing pipeline without OCR. Board images will still be displayed.$(RESET)"; \
				echo "$(BOLD)$(YELLOW)You can try again later with: make ocr_boards TITLE=\"$(TITLE)\"$(RESET)"; \
			}; \
		fi; \
	fi

# Extract transcript from video
extract_transcript:
	@if [ -z "$(VIDEO)" ]; then \
		echo "$(BOLD)$(YELLOW)Error: VIDEO parameter is required$(RESET)"; \
		echo "Usage: make extract_transcript VIDEO=path/to/video.mp4 TITLE=\"Lecture Title\""; \
		exit 1; \
	fi
	@echo "$(BOLD)$(BLUE)Extracting transcript from video: $(VIDEO)$(RESET)"
	@echo "$(BOLD)$(BLUE)Title: $(TITLE)$(RESET)"
	@mkdir -p $(OUTPUT_DIR)
	@if [ "$(USE_GROQ)" = "true" ]; then \
		echo "Using Groq API for enhanced transcription..."; \
		python get_transcript.py "$(VIDEO)" --use-groq --out $(TRANSCRIPT_JSON); \
	else \
		echo "Using local Whisper model for transcription..."; \
		python get_transcript.py "$(VIDEO)" --out $(TRANSCRIPT_JSON); \
	fi

# Copy output to dashboard directory
dashboard_copy:
	@if [ ! -d $(OUTPUT_DIR) ]; then \
		echo "$(BOLD)$(YELLOW)Error: Output directory not found. Run processing steps first.$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BOLD)$(BLUE)Setting up dashboard with processed content for: $(TITLE)$(RESET)"
	@mkdir -p "$(DASHBOARD_OUTPUT)/boards"
	@if [ -f $(TRANSCRIPT_JSON) ]; then \
		cp $(TRANSCRIPT_JSON) "$(DASHBOARD_OUTPUT)/"; \
		echo "✅ Copied transcript.json to dashboard"; \
	else \
		echo "$(YELLOW)Warning: Transcript file not found. Skipping...$(RESET)"; \
	fi
	@if [ -f $(BOARDS_JSON) ]; then \
		cp $(BOARDS_JSON) "$(DASHBOARD_OUTPUT)/boards/"; \
		echo "✅ Copied boards.json to dashboard"; \
		if [ -d $(BOARDS_DIR) ]; then \
			cp $(BOARDS_DIR)/*.jpg "$(DASHBOARD_OUTPUT)/boards/" 2>/dev/null || echo "No board images found."; \
			echo "✅ Copied board images to dashboard"; \
		fi; \
	else \
		echo "$(YELLOW)Warning: Boards file not found. Skipping...$(RESET)"; \
	fi

# Launch the dashboard
launch_dashboard:
	@echo "$(BOLD)$(GREEN)Setting up and launching dashboard...$(RESET)"
	@cd dashboard && npm install
	@echo "$(BOLD)$(GREEN)Dashboard ready! Starting the development server...$(RESET)"
	@echo "$(YELLOW)Open http://localhost:5173 in your browser to view the dashboard$(RESET)"
	@cd dashboard && npm run dev

# -------- Utility targets ----------------------------------------------------

# Clean generated files
clean:
	@echo "$(BOLD)$(BLUE)Cleaning generated files for: $(TITLE)$(RESET)"
	@if [ -d $(OUTPUT_DIR) ]; then \
		rm -rf $(OUTPUT_DIR); \
		echo "✅ Removed output directory"; \
	else \
		echo "$(YELLOW)Output directory not found. Nothing to clean.$(RESET)"; \
	fi
	@if [ -d "$(DASHBOARD_OUTPUT)/boards" ]; then \
		rm -rf "$(DASHBOARD_OUTPUT)/boards"; \
		echo "✅ Removed dashboard board images"; \
	fi
	@if [ -f "$(DASHBOARD_OUTPUT)/transcript.json" ]; then \
		rm -f "$(DASHBOARD_OUTPUT)/transcript.json"; \
		echo "✅ Removed dashboard transcript"; \
	fi
	@echo "$(GREEN)Cleanup complete!$(RESET)"

# Install dependencies
install_deps:
	@echo "$(BOLD)$(BLUE)Installing Python dependencies...$(RESET)"
	pip install -r requirements.txt
	@echo "$(BOLD)$(BLUE)Installing Node.js dependencies for dashboard...$(RESET)"
	cd dashboard && npm install
	@echo "$(GREEN)All dependencies installed successfully!$(RESET)"

# -------- Documentation ------------------------------------------------------

# Help target with enhanced documentation
help:
	@echo "$(BOLD)✨ LecToNotes Makefile - Documentation ✨$(RESET)"
	@echo ""
	@echo "$(BOLD)OVERVIEW:$(RESET)"
	@echo "LecToNotes extracts content from math lecture videos, processing both"
	@echo "the audio transcript and board content for structured viewing."
	@echo ""
	@echo "$(BOLD)MAIN COMMANDS:$(RESET)"
	@echo "  $(BOLD)make process_video VIDEO=path/to/video.mp4 TITLE=\"Lecture Title\"$(RESET)"
	@echo "      Complete end-to-end processing of a video lecture, including dashboard setup"
	@echo ""
	@echo "$(BOLD)SETUP:$(RESET)"
	@echo "  $(BOLD)make install_deps$(RESET)           Install all required dependencies"
	@echo ""
	@echo "$(BOLD)INDIVIDUAL PIPELINE STEPS:$(RESET)"
	@echo "  $(BOLD)make extract_transcript VIDEO=path.mp4 TITLE=\"Title\"$(RESET)"
	@echo "      Extract transcript from video using Whisper"
	@echo ""
	@echo "  $(BOLD)make extract_boards VIDEO=path.mp4 TITLE=\"Title\"$(RESET)"
	@echo "      Extract board images from video using computer vision"
	@echo ""
	@echo "  $(BOLD)make ocr_boards TITLE=\"Title\"$(RESET)"
	@echo "      Process board images with OCR using LLMs"
	@echo ""
	@echo "  $(BOLD)make dashboard_copy TITLE=\"Title\"$(RESET)"
	@echo "      Copy output files to dashboard for viewing"
	@echo ""
	@echo "  $(BOLD)make launch_dashboard$(RESET)"
	@echo "      Start the dashboard development server"
	@echo ""
	@echo "  $(BOLD)make clean TITLE=\"Title\"$(RESET)"
	@echo "      Remove all generated files for a specific lecture"
	@echo ""
	@echo "$(BOLD)PARAMETERS:$(RESET)"
	@echo "  $(BOLD)VIDEO$(RESET)                Video file path or YouTube URL"
	@echo "  $(BOLD)TITLE$(RESET)                Lecture title (default: \"Untitled Lecture\")"
	@echo "  $(BOLD)INTERVAL$(RESET)             Time interval for segmentation in seconds (default: 300)"
	@echo "  $(BOLD)USE_GROQ$(RESET)             Use Groq API for transcription (true/false, default: false)"
	@echo "  $(BOLD)SKIP_HUMANS$(RESET)          Skip human detection in board extraction (true/false, default: false)"
	@echo "  $(BOLD)HUMAN_CONFIDENCE$(RESET)     Confidence threshold for human detection (0.0-1.0, default: 0.4)"
	@echo "  $(BOLD)SKIP_OCR$(RESET)             Skip OCR processing (true/false, default: false)"
	@echo ""
	@echo "$(BOLD)EXAMPLES:$(RESET)"
	@echo "  Process a local video:"
	@echo "    $(BOLD)make process_video VIDEO=path/to/video.mp4 TITLE=\"Linear Algebra Lecture 1\"$(RESET)"
	@echo ""
	@echo "  Process a YouTube video with Groq transcription:"
	@echo "    $(BOLD)make process_video VIDEO=\"https://youtube.com/watch?v=VIDEO_ID\" TITLE=\"Calculus\" USE_GROQ=true$(RESET)"
	@echo ""
	@echo "  Process video but skip OCR (if you don't have an OpenRouter API key):"
	@echo "    $(BOLD)make process_video VIDEO=lecture.mp4 TITLE=\"Physics\" SKIP_OCR=true$(RESET)"
	@echo ""
	@echo "  Process video but include lecturer in board captures:"
	@echo "    $(BOLD)make process_video VIDEO=lecture.mp4 TITLE=\"Chemistry\" SKIP_HUMANS=true$(RESET)"
	@echo ""
	@echo "  Adjust human detection sensitivity:"
	@echo "    $(BOLD)make process_video VIDEO=lecture.mp4 TITLE=\"Algebra\" HUMAN_CONFIDENCE=0.6$(RESET)"
	@echo ""
	@echo "  Process each step individually:"
	@echo "    $(BOLD)make extract_transcript VIDEO=lecture.mp4 TITLE=\"Physics\"$(RESET)"
	@echo "    $(BOLD)make extract_boards VIDEO=lecture.mp4 TITLE=\"Physics\"$(RESET)"
	@echo "    $(BOLD)make ocr_boards TITLE=\"Physics\"$(RESET)"
	@echo "    $(BOLD)make dashboard_copy TITLE=\"Physics\"$(RESET)"
	@echo "    $(BOLD)make launch_dashboard$(RESET)"
	@echo ""
	@echo "$(BOLD)REQUIREMENTS:$(RESET)"
	@echo "  - Python 3.8+ with dependencies from requirements.txt"
	@echo "  - Node.js 18+ for the dashboard"
	@echo "  - API keys in .env file:"
	@echo "    $(BOLD)GROQ_API_KEY$(RESET)       Required for enhanced transcription (optional)"
	@echo "    $(BOLD)OPENROUTER_API_KEY$(RESET) Required for OCR processing"
	@echo ""
	@echo "$(BOLD)TROUBLESHOOTING:$(RESET)"
	@echo "  - $(BOLD)401 Unauthorized for OpenRouter API:$(RESET)"
	@echo "    - Verify your OpenRouter API key in .env file"
	@echo "    - Sign up at https://openrouter.ai if you don't have an account"
	@echo "    - Run with SKIP_OCR=true to bypass OCR: make process_video VIDEO=file.mp4 SKIP_OCR=true"
	@echo ""
	@echo "  - $(BOLD)Human detection issues:$(RESET)"
	@echo "    - If the system removes too many boards with the lecturer: lower HUMAN_CONFIDENCE to 0.3"
	@echo "    - If the system doesn't filter out enough lecturer images: increase HUMAN_CONFIDENCE to 0.6" 
	@echo "    - If you want to include frames with the lecturer, use: SKIP_HUMANS=true"
	@echo "    - Example: make extract_boards VIDEO=file.mp4 TITLE=\"Title\" HUMAN_CONFIDENCE=0.3"
	@echo ""
	@echo "  - $(BOLD)Incomplete results:$(RESET)"
	@echo "    - Each step can be run individually if a previous step failed"
	@echo "    - Check output directories for partial results"
	@echo "    - Look in the output/boards/ocr/ directory for OCR-optimized images"

.PHONY: process_video extract_boards ocr_boards extract_transcript dashboard_copy launch_dashboard clean install_deps help