import os
import cv2
import argparse
import json
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm  # Add progress bar support

# Import TensorFlow dependencies - required
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_preprocess

# pHash computation
def phash(image, size=32, smaller_size=8):
    img = cv2.resize(image, (size, size)).astype(np.float32)
    dct = cv2.dct(img)
    dct_low = dct[:smaller_size, :smaller_size]
    med = np.median(dct_low)
    return (dct_low > med).flatten()

# Optical flow mean magnitude
def mean_flow(prev, curr):
    flow = cv2.calcOpticalFlowFarneback(prev, curr, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
    return np.mean(mag)

# Detect largest rectangular board in a frame, return crop or None
def detect_board(frame):
    if frame is None or frame.size == 0:
        return None
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    # Try multiple edge detection parameters for better detection
    edges_list = [
        cv2.Canny(blur, 50, 150),  # Default
        cv2.Canny(blur, 30, 100)   # More sensitive for low contrast
    ]
    
    best_boards = []
    
    for edges in edges_list:
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            if len(approx) == 4 and cv2.isContourConvex(approx):
                x, y, w, h = cv2.boundingRect(approx)
                area = w * h
                ar = w / float(h)
                
                if area > 150000 and 1.0 < ar < 3.5:
                    best_boards.append((area, (x, y, w, h)))
    
    if not best_boards:
        return None
    
    # Use the board with largest area
    best_boards.sort(key=lambda b: b[0], reverse=True)
    x, y, w, h = best_boards[0][1]
    
    # Check if board is empty (mostly uniform color)
    board_crop = frame[y:y+h, x:x+w]
    gray_crop = gray[y:y+h, x:x+w]
    
    # Simple check for blank boards - very low variance
    laplacian = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
    if laplacian < 50.0:  # Very low texture
        return None  # Skip blank boards
    
    return board_crop, gray_crop

# Function to detect blank boards - to be used after deduplication
def is_blank_board(img, threshold=0.95):
    """
    Detect if a board image is blank (no writing) based on histogram and edge analysis.
    
    Args:
        img: Input image (color or grayscale)
        threshold: Threshold for blank detection (higher = more strict)
        
    Returns:
        boolean: True if board is blank, False otherwise
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # 1. Edge detection approach
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = cv2.countNonZero(edges) / (gray.size)
    
    # If very few edges detected, likely blank
    if edge_ratio < 0.02:
        return True
    
    # 2. Otsu's thresholding to separate foreground/background
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Calculate the percentage of background pixels
    if np.mean(thresh) > 127:  # If background is white
        bg_ratio = cv2.countNonZero(thresh) / thresh.size
    else:  # If background is black
        bg_ratio = 1 - (cv2.countNonZero(thresh) / thresh.size)
    
    # If background occupies most of the image (with some tolerance), consider it blank
    return bg_ratio > threshold

# Main pipeline
def main():
    parser = argparse.ArgumentParser(description="Extract and deduplicate blackboard images from lecture video.")
    parser.add_argument("-i", "--input", required=True, help="Path to input .mp4 video file")
    parser.add_argument("-o", "--output", required=True, help="Directory to save final images and metadata")
    parser.add_argument("--interval", type=float, default=2.5, 
                      help="Sampling interval in seconds (default: 2.5)")
    parser.add_argument("--quality", type=int, default=90,
                      help="JPEG quality for saved images (0-100, default: 90)")
    parser.add_argument("--keep-similar", action="store_true",
                      help="Keep more similar boards to avoid missing content")
    args = parser.parse_args()

    video_path = args.input
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    # Step 1: Extract board crops with timestamps
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = int(fps * args.interval)  # Convert seconds to frames
    print(f"Processing video: {video_path}")
    print(f"Sampling every {args.interval} seconds ({interval} frames)")
    
    # Calculate total number of frames to sample for progress bar
    total_samples = total_frames // interval + 1
    
    boards = []  # list of dicts: {'timestamp': , 'orig': img, 'gray_small': }
    frame_num = 0
    
    # Progress bar for frame extraction
    with tqdm(total=total_samples, desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_num % interval == 0:
                timestamp = frame_num / fps
                result = detect_board(frame)
                
                if result:
                    crop, gray_crop = result
                    gray_small = cv2.resize(gray_crop, (300, 200))
                    boards.append({
                        'timestamp': timestamp, 
                        'orig': crop, 
                        'gray_small': gray_small,
                        'frame_num': frame_num
                    })
                pbar.update(1)
            
            frame_num += 1
    
    cap.release()
    print(f"Extracted {len(boards)} candidate boards from {frame_num} frames")

    # Step 2: Prepare metrics for deduplication
    print("Computing similarity metrics...")
    for b in tqdm(boards, desc="Computing metrics"):
        # Laplacian variance (sharpness)
        b['sharpness'] = cv2.Laplacian(b['gray_small'], cv2.CV_64F).var()
        # Perceptual hash
        b['phash'] = phash(b['gray_small'])

    # Compute deep CNN embeddings for better similarity comparison
    print("Computing deep embeddings with CNN...")
    # Prefer MobileNetV2 as it's smaller and faster
    if 'MobileNetV2' in globals():
        base = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
        preprocess = mobilenet_preprocess
    else:
        base = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        preprocess = resnet_preprocess
        
    size_model = (224, 224)
    
    def embed(img):
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, size_model)
        x = np.expand_dims(x, 0)
        x = preprocess(x)
        return base.predict(x, verbose=0).flatten()
        
    for b in tqdm(boards, desc="CNN embedding"):
        b['deep'] = embed(b['orig'])
        
    # Use CNN-based similarity for deduplication - less strict for keep-similar mode
    deep_thresh = 0.85 if args.keep_similar else 0.9

    # Deduplication parameters - less strict to avoid missing content
    phash_thresh = 25 if args.keep_similar else 18  # Higher threshold = more boards
    ssim_thresh = 0.75 if args.keep_similar else 0.80  # Lower threshold = more boards

    # Step 3: Temporal grouping - identify segments with different content
    print("Performing temporal segmentation...")
    segments = []
    if boards:
        prev_gray = boards[0]['gray_small']
        segments.append([])
        
        for b in tqdm(boards, desc="Temporal grouping"):
            curr_gray = b['gray_small']
            
            # Measure similarity using multiple methods
            score = ssim(prev_gray, curr_gray)
            flow_val = mean_flow(prev_gray, curr_gray)
            
            # Start a new segment if there's a significant change - less strict threshold
            if score < 0.85 or flow_val > 1.5:  # Was 0.90 and 2.0
                segments.append([])
            
            segments[-1].append(b)
            prev_gray = curr_gray
    
    # Keep top N frames from each segment based on quality, not just the best one
    pre_candidates = []
    for segment in segments:
        if segment:
            # Sort by sharpness and keep the top frames
            sorted_segment = sorted(segment, key=lambda x: x['sharpness'], reverse=True)
            # Take top 2 frames if keep-similar mode is on and segment has enough frames
            if args.keep_similar and len(sorted_segment) >= 2:
                pre_candidates.extend(sorted_segment[:2])
            else:
                pre_candidates.append(sorted_segment[0])
    
    print(f"Identified {len(segments)} distinct content segments")
    print(f"Selected {len(pre_candidates)} candidates after temporal grouping")

    # Step 4: Group similar boards using perceptual hashing
    print("Clustering similar boards...")
    clusters = []
    for b in tqdm(pre_candidates, desc="Clustering"):
        placed = False
        for c in clusters:
            # Perceptual hash distance
            dist = np.count_nonzero(c[0]['phash'] != b['phash'])
            if dist <= phash_thresh:
                c.append(b)
                placed = True
                break
        
        if not placed:
            clusters.append([b])
    
    print(f"Clustered into {len(clusters)} content groups")

    # Step 5: Fine-grained deduplication within each cluster
    print("Performing final selection...")
    final = []
    for cluster in tqdm(clusters, desc="Final selection"):
        subclusters = []
        for b in cluster:
            added = False
            for sc in subclusters:
                rep = sc[0]
                
                # Check CNN similarity
                cos = np.dot(rep['deep'], b['deep']) / (np.linalg.norm(rep['deep']) * np.linalg.norm(b['deep']) + 1e-10)
                cnn_ok = cos >= deep_thresh
                
                # Also check structural similarity
                ssim_ok = ssim(rep['gray_small'], b['gray_small']) >= ssim_thresh
                
                # Add to existing subcluster if similar
                if ssim_ok and cnn_ok:
                    sc.append(b)
                    added = True
                    break
            
            # Create new subcluster if not added to any existing one
            if not added:
                subclusters.append([b])
        
        # Choose best quality representative from each subcluster
        for sc in subclusters:
            best = max(sc, key=lambda x: x['sharpness'])
            final.append(best)
    
    print(f"Selected {len(final)} boards after deduplication")
    
    # Additional post-processing to remove blank boards
    print("Filtering out blank boards...")
    non_blank_boards = []
    blank_count = 0
    
    for b in tqdm(final, desc="Filtering blanks"):
        if not is_blank_board(b['orig'], threshold=0.98):  # Higher threshold to keep more boards
            non_blank_boards.append(b)
        else:
            blank_count += 1
    
    print(f"Removed {blank_count} blank boards")
    final = non_blank_boards
    
    print(f"Final selection: {len(final)} unique, non-blank boards")

    # Step 6: Save results and metadata with timestamp-based filenames
    print("Saving images and metadata...")
    meta = []
    
    # Create OCR directory to maintain compatibility with the pipeline
    ocr_dir = os.path.join(out_dir, "ocr")
    os.makedirs(ocr_dir, exist_ok=True)

    for b in tqdm(sorted(final, key=lambda x: x['timestamp']), desc="Saving images"):
        # Generate timestamp components for filename
        total_ms = int(b['timestamp'] * 1000)
        hrs = total_ms // 3600000
        mins = (total_ms % 3600000) // 60000
        secs = (total_ms % 60000) // 1000
        ms = total_ms % 1000
        
        # Create filename and path
        fname = f"board_{hrs:02d}_{mins:02d}_{secs:02d}_{ms:03d}.jpg"
        path = os.path.join(out_dir, fname)
        
        # Save image with specified quality
        cv2.imwrite(path, b['orig'], [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
        
        # Save a copy in the OCR directory with proper preprocessing for LLM OCR
        ocr_fname = f"ocr_{hrs:02d}_{mins:02d}_{secs:02d}_{ms:03d}.jpg"
        ocr_path = os.path.join(ocr_dir, ocr_fname)
        
        # Improved OCR preprocessing that avoids the white splash issue
        if len(b['orig'].shape) == 3:
            gray = cv2.cvtColor(b['orig'], cv2.COLOR_BGR2GRAY)
        else:
            gray = b['orig'].copy()
        
        # Apply CLAHE to enhance contrast while preserving details
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Use adaptive thresholding instead of global Otsu to avoid large white regions
        binary = cv2.adaptiveThreshold(
            enhanced, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            21,  # Block size
            5    # Constant subtracted from mean
        )
        
        # Mild denoising with small kernel
        kernel = np.ones((2, 2), np.uint8)
        denoised = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Apply very mild Gaussian blur (3x3 kernel) to reduce noise while preserving text
        blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
        
        # Save preprocessed image for OCR
        cv2.imwrite(ocr_path, blurred, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
        
        # Format timestamp for metadata
        formatted_timestamp = f"{hrs:02d}_{mins:02d}_{secs:02d}"
        
        # Use relative path for compatibility with other tools
        rel_path = os.path.basename(path)
        meta.append({'timestamp': formatted_timestamp, 'path': rel_path})

    # Save metadata to JSON
    with open(os.path.join(out_dir, 'boards.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Extracted and deduplicated {len(final)} boards. Metadata saved to boards.json")
    print(f"✅ OCR optimized boards saved to {ocr_dir}")
    print(f"Tip: Run with --keep-similar for even more boards if needed")

if __name__ == '__main__':
    main()