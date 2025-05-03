# LecToNotes Makefile - Enhanced for complete video processing pipeline

# Default values
VIDEO ?= 
TITLE ?= "Untitled Lecture"
INTERVAL ?= 300
USE_GROQ ?= false

# Directories
OUTPUT_DIR = "$(TITLE) OUTPUT"
BOARDS_DIR = $(OUTPUT_DIR)/boards
DASHBOARD_OUTPUT = dashboard/public/OUTPUT

# Output files
BOARDS_JSON = $(BOARDS_DIR)/boards.json
TRANSCRIPT_JSON = $(OUTPUT_DIR)/transcript.json

# Colors for prettier output
BOLD = \033[1m
GREEN = \033[32m
BLUE = \033[34m
YELLOW = \033[33m
RESET = \033[0m

# Main target - process a video completely with dashboard setup
process_video: extract_transcript extract_boards ocr_boards dashboard_copy launch_dashboard

# Extract board images from video
extract_boards:
	@echo "$(BOLD)$(BLUE)Extracting board images from video...$(RESET)"
	@mkdir -p $(BOARDS_DIR)
	python get_boards.py -i $(VIDEO) -o $(BOARDS_DIR) --preoptimize-ocr --enhance

# Process board images with OCR
ocr_boards:
	@echo "$(BOLD)$(BLUE)Processing board images with OCR...$(RESET)"
	python LLM_OCR.py $(BOARDS_JSON) --transcript $(TRANSCRIPT_JSON)

# Extract transcript from video
extract_transcript:
	@echo "$(BOLD)$(BLUE)Extracting transcript from video...$(RESET)"
	@mkdir -p $(OUTPUT_DIR)
	if [ "$(USE_GROQ)" = "true" ]; then \
		python get_transcript.py $(VIDEO) --use-groq --out $(TRANSCRIPT_JSON); \
	else \
		python get_transcript.py $(VIDEO) --out $(TRANSCRIPT_JSON); \
	fi

# Copy output to dashboard directory
dashboard_copy:
	@echo "$(BOLD)$(BLUE)Setting up dashboard with processed content...$(RESET)"
	@mkdir -p $(DASHBOARD_OUTPUT)/boards
	@cp $(TRANSCRIPT_JSON) $(DASHBOARD_OUTPUT)/
	@echo "✓ Copied transcript.json to dashboard"
	@cp $(BOARDS_JSON) $(DASHBOARD_OUTPUT)/boards/
	@echo "✓ Copied boards.json to dashboard"
	@cp $(BOARDS_DIR)/*.jpg $(DASHBOARD_OUTPUT)/boards/
	@echo "✓ Copied board images to dashboard"

# Launch the dashboard
launch_dashboard:
	@echo "$(BOLD)$(GREEN)Setting up and launching dashboard...$(RESET)"
	@cd dashboard && npm install
	@echo "$(BOLD)$(GREEN)Dashboard ready! Starting the development server...$(RESET)"
	@echo "$(YELLOW)Open http://localhost:5173 in your browser to view the dashboard$(RESET)"
	@cd dashboard && npm run dev

# Clean generated files
clean:
	@echo "$(BOLD)$(BLUE)Cleaning generated files...$(RESET)"
	rm -rf $(OUTPUT_DIR)
	rm -rf $(DASHBOARD_OUTPUT)/boards
	rm -f $(DASHBOARD_OUTPUT)/transcript.json
	@echo "$(GREEN)Cleanup complete!$(RESET)"

# Install dependencies
install_deps:
	@echo "$(BOLD)$(BLUE)Installing Python dependencies...$(RESET)"
	pip install -r requirements.txt
	@echo "$(BOLD)$(BLUE)Installing Node.js dependencies for dashboard...$(RESET)"
	cd dashboard && npm install
	@echo "$(GREEN)All dependencies installed successfully!$(RESET)"

# Help target with enhanced documentation
help:
	@echo "$(BOLD)✨ LecToNotes Makefile - Enhanced Documentation ✨$(RESET)"
	@echo ""
	@echo "$(BOLD)MAIN COMMANDS:$(RESET)"
	@echo "  $(BOLD)make process_video VIDEO=path/to/video.mp4 TITLE=\"Lecture Title\"$(RESET)"
	@echo "      Complete end-to-end processing of a video lecture, including dashboard setup"
	@echo ""
	@echo "$(BOLD)SETUP:$(RESET)"
	@echo "  $(BOLD)make install_deps$(RESET)           Install all required dependencies"
	@echo ""
	@echo "$(BOLD)INDIVIDUAL PIPELINE STEPS:$(RESET)"
	@echo "  $(BOLD)make extract_transcript$(RESET)     Extract transcript from video"
	@echo "  $(BOLD)make extract_boards$(RESET)        Extract board images from video"
	@echo "  $(BOLD)make ocr_boards$(RESET)            Process board images with OCR"
	@echo "  $(BOLD)make dashboard_copy$(RESET)        Copy output files to dashboard"
	@echo "  $(BOLD)make launch_dashboard$(RESET)      Start the dashboard development server"
	@echo "  $(BOLD)make clean$(RESET)                 Remove all generated files"
	@echo ""
	@echo "$(BOLD)PARAMETERS:$(RESET)"
	@echo "  $(BOLD)VIDEO$(RESET)                Video file path or YouTube URL"
	@echo "  $(BOLD)TITLE$(RESET)                Lecture title (default: \"Untitled Lecture\")"
	@echo "  $(BOLD)INTERVAL$(RESET)             Time interval for segmentation in seconds (default: 300)"
	@echo "  $(BOLD)USE_GROQ$(RESET)             Use Groq API for transcription (true/false, default: false)"
	@echo ""
	@echo "$(BOLD)EXAMPLES:$(RESET)"
	@echo "  Process a local video:"
	@echo "    $(BOLD)make process_video VIDEO=/path/to/video.mp4 TITLE=\"Linear Algebra Lecture 1\"$(RESET)"
	@echo ""
	@echo "  Process a YouTube video with Groq transcription:"
	@echo "    $(BOLD)make process_video VIDEO=\"https://youtube.com/watch?v=VIDEO_ID\" TITLE=\"Calculus\" USE_GROQ=true$(RESET)"
	@echo ""
	@echo "$(BOLD)REQUIREMENTS:$(RESET)"
	@echo "  - Python 3.8+ with dependencies from requirements.txt"
	@echo "  - Node.js 18+ for the dashboard"
	@echo "  - API keys in .env file (GROQ_API_KEY, OPENROUTER_API_KEY)"

.PHONY: process_video extract_boards ocr_boards extract_transcript dashboard_copy launch_dashboard clean install_deps help