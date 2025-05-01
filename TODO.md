# LecToNotes Development Plan

## 1. Enhanced Content Navigation

- **Timeline/Progress Bar**
  - Implement a visual timeline showing lecture segments
  - Add clickable markers where board content changes
  - Include timestamp indicators and hover previews

- **Keyboard Navigation**
  - Add keyboard shortcuts (left/right arrows) for jumping between segments
  - Implement time-based jumps (e.g., 30s, 1min, 5min)
  - Add quick-jump to specific timestamps

- **Table of Contents View**
  - Extract section headings from board content
  - Create collapsible sidebar navigation panel
  - Link TOC entries to specific timestamps

## 2. Display Mathematics Functionality

- **LaTeX Rendering Improvements**
  - Fix current rendering issues by implementing KaTeX or MathJax for proper LaTeX display
  - Create custom renderer to handle specialized math notation
  - Implement syntax highlighting for mathematical expressions
  - Add options to adjust rendering style/size

- **OCR Enhancement** (`LLM_OCR.py`)
  - Refine prompt engineering to prioritize accurate LaTeX extraction
  - Implement multi-pass OCR with specialized math symbol detection
  - Add post-processing rules specific to common math notation patterns
  - Create validation step to ensure extracted LaTeX is syntactically correct
  - Consider fine-tuning OCR model specifically for mathematical notation

- **Board Extraction Improvements** (`get_boards.py`)
  - Enhance board detection algorithm for better edge detection
  - Implement perspective correction for angled board shots
  - Add contrast/brightness normalization for better image quality
  - Refine deduplication algorithm with better thresholds
  - Consider adding options for manual board selection/correction

- **Markdown Integration**
  - Properly fence LaTeX blocks for correct rendering
  - Create hybrid Markdown+LaTeX renderer
  - Implement proper escaping for LaTeX within Markdown

## 3. Image Extraction Improvements

- **Frame Selection Optimization**
  - Refine sampling interval logic (currently every 3 seconds) to adapt based on content change rate
  - Add scene detection to identify when instructor moves away from/to the board
  - Implement multi-resolution analysis to better identify frames with clear, readable content

- **Enhanced Deduplication**
  - Tune threshold parameters (`phash_thresh`, `ssim_thresh`, `deep_thresh`) for better discrimination
  - Add content-aware filtering to prioritize frames with more text/equations
  - Implement weighted scoring system combining image quality metrics with content density
  - Add temporal importance weighting (e.g., prefer frames that stay visible longer)
  - Consider hierarchical clustering approach for more precise similarity grouping

- **Image Quality Improvements**
  - Add intelligent contrast enhancement for better readability
  - Implement glare/reflection removal for boards with lighting issues
  - Add image sharpening specific to handwritten content
  - Create image segmentation to isolate and enhance text regions
  - Consider super-resolution for low-quality video sources

## 4. UI/UX Improvements

- **Advanced Image Viewing**
  - Implement zoom/pan controls for board images
  - Add image comparison view (before/after)
  - Create fullscreen viewing mode for detailed examination

- **Dark Mode**
  - Implement system-preference-based theme detection
  - Create dark color scheme for all UI elements
  - Ensure proper contrast for mathematical expressions

- **Layout Improvements**
  - Create resizable panels for transcript/board sections
  - Add options to hide/show different panels
  - Implement responsive design (prioritizing desktop)

## 5. Export & Integration Features (Low Priority)

- **Export Options**
  - Add PDF export with proper LaTeX rendering
  - Create Markdown export for note-taking applications
  - Implement structured JSON export for programmatic use

- **Note-Taking**
  - Add simple note-taking capability alongside lecture content
  - Create timestamp-anchored annotations
  - Implement basic formatting options for notes
