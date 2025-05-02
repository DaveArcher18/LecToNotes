import os
import cv2
import argparse
import json
import numpy as np
import pickle
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# Optional deep-feature embeddings
try:
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_preprocess
    tf_available = True
except ImportError:
    tf_available = False

# Optional: Display progress bars if tqdm is available
try:
    from tqdm import tqdm
    tqdm_available = True
except ImportError:
    tqdm_available = False
    tqdm = lambda x, **kwargs: x  # Fallback to identity function

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

# Enhanced board detection with perspective correction
def detect_board(frame, min_area=150000, aspect_ratio_range=(1.0, 3.5)):
    if frame is None or frame.size == 0:
        return None
    
    # Convert to grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()
    
    # Apply adaptive thresholding
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Try multiple edge detection params to get the best contours
    best_boards = []
    
    # Try different parameters to get better edges
    edge_params = [
        (50, 150),  # Default
        (30, 100),  # More sensitive
        (80, 200)   # Less sensitive
    ]
    
    for low, high in edge_params:
        edges = cv2.Canny(blur, low, high)
        
        # Dilate to connect edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            # Check if it's a quadrilateral and convex
            if len(approx) == 4 and cv2.isContourConvex(approx):
                x, y, w, h = cv2.boundingRect(approx)
                area = w * h
                ar = w / float(h)
                
                # Filter by area and aspect ratio
                if area > min_area and aspect_ratio_range[0] < ar < aspect_ratio_range[1]:
                    best_boards.append((approx, area, (x, y, w, h)))
    
    if not best_boards:
        return None
    
    # Sort by area (largest first)
    best_boards.sort(key=lambda b: b[1], reverse=True)
    
    # Try perspective transform with the largest board
    quad_points = best_boards[0][0].reshape(4, 2).astype(np.float32)
    
    # Sort points to ensure consistent order: top-left, top-right, bottom-right, bottom-left
    # First sort by y-coordinate (top to bottom)
    sorted_by_y = quad_points[np.argsort(quad_points[:, 1])]
    # Get top and bottom points
    top_points = sorted_by_y[:2]
    bottom_points = sorted_by_y[2:]
    # Sort top points by x (left to right)
    top_points = top_points[np.argsort(top_points[:, 0])]
    # Sort bottom points by x (left to right)
    bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]
    # Combine points in order: top-left, top-right, bottom-right, bottom-left
    ordered_points = np.vstack((top_points, bottom_points[::-1]))
    
    # Get width and height for the destination image
    width = int(max(
        np.linalg.norm(ordered_points[0] - ordered_points[1]),
        np.linalg.norm(ordered_points[2] - ordered_points[3])
    ))
    height = int(max(
        np.linalg.norm(ordered_points[0] - ordered_points[3]),
        np.linalg.norm(ordered_points[1] - ordered_points[2])
    ))
    
    # Define destination points
    dst_points = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)
    
    # Calculate perspective transform matrix
    matrix = cv2.getPerspectiveTransform(ordered_points, dst_points)
    
    # Apply perspective transformation
    try:
        warped = cv2.warpPerspective(frame, matrix, (width, height))
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else warped
        return warped, warped_gray
    except cv2.error:
        # Fallback to simple crop if perspective transform fails
        x, y, w, h = best_boards[0][2]
        crop = frame[y:y+h, x:x+w]
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else crop
        return crop, gray_crop

# Enhance image quality for better OCR
def enhance_image(img):
    # Skip if image is None or empty
    if img is None or img.size == 0:
        return img
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Convert back to color if input was color
    if len(img.shape) == 3:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    return enhanced

# Compute frame sampling based on video length
def compute_sampling_interval(cap, target_frames=200, min_interval=3, max_interval=15):
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    # Compute an interval that would yield approximately target_frames
    # but keep it between min_interval and max_interval seconds
    video_length_sec = total_frames / fps
    optimal_interval_sec = video_length_sec / target_frames
    
    # Clamp the interval
    interval_sec = max(min_interval, min(max_interval, optimal_interval_sec))
    
    # Convert to frames
    interval_frames = int(interval_sec * fps)
    
    return interval_frames

# Optimize image for OCR (similar to LLM_OCR.py's optimize_image_for_ocr)
def optimize_for_ocr(img):
    """Optimize image for OCR processing, making it compatible with LLM_OCR.py's expectations."""
    if img is None or img.size == 0:
        return img
    
    # Resize with aspect ratio preservation
    max_dim = 1024
    h, w = img.shape[:2]
    if h > w:
        new_h, new_w = max_dim, int(max_dim * w / h)
    else:
        new_h, new_w = int(max_dim * h / w), max_dim
    img = cv2.resize(img, (new_w, new_h))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to better handle chalk/marker on blackboard
    # This helps with contrast differences across the board
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, 15
    )
    
    # Morphological operations to clean up text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Convert back to normal polarity
    morph = cv2.bitwise_not(morph)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Create a color image that highlights the text
    # Blend the enhanced image with the original
    final = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    # Sharpen the image using unsharp mask
    blurred = cv2.GaussianBlur(final, (0, 0), 3)
    sharpened = cv2.addWeighted(final, 1.5, blurred, -0.5, 0)
    
    # Apply a slight bilateral filter to remove noise while preserving edges
    final = cv2.bilateralFilter(sharpened, 9, 75, 75)
    
    return final

# Main pipeline
def main():
    parser = argparse.ArgumentParser(description="Extract and deduplicate blackboard images from lecture video.")
    parser.add_argument("-i", "--input", required=True, help="Path to input .mp4 video file")
    parser.add_argument("-o", "--output", required=True, help="Directory to save final images and metadata")
    parser.add_argument("--quality", choices=["high", "medium", "low"], default="medium", 
                      help="Output image quality (defaults to medium)")
    parser.add_argument("--min-area", type=int, default=150000, 
                      help="Minimum area for board detection (default: 150000)")
    parser.add_argument("--sampling-rate", type=float, default=None,
                      help="Frame sampling rate in seconds (default: auto)")
    parser.add_argument("--enhance", action="store_true", 
                      help="Apply image enhancement for better OCR")
    parser.add_argument("--preoptimize-ocr", action="store_true",
                      help="Pre-optimize images for LLM_OCR.py and save in a pickle file")
    args = parser.parse_args()

    video_path = args.input
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    print(f"✓ Processing video: {os.path.basename(video_path)}")
    print(f"✓ Output directory: {out_dir}")

    # Quality settings
    if args.quality == "high":
        jpeg_quality = 95
        ssim_thresh = 0.85
        phash_thresh = 15
    elif args.quality == "medium":
        jpeg_quality = 90
        ssim_thresh = 0.80
        phash_thresh = 20
    else:  # low
        jpeg_quality = 85
        ssim_thresh = 0.75
        phash_thresh = 25

    # Step 1: Extract board crops with timestamps
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Compute optimal sampling interval
    if args.sampling_rate:
        interval = int(fps * args.sampling_rate)
    else:
        interval = compute_sampling_interval(cap)
    
    est_samples = total_frames // interval
    print(f"✓ Sampling every {interval} frames (~{interval/fps:.1f} seconds)")
    print(f"✓ Estimated samples: {est_samples}")
    
    boards = []  # list of dicts: {'timestamp': , 'orig': img, 'gray_small': }
    
    # Process frames with progress bar
    frame_num = 0
    pbar = tqdm(total=total_frames//interval, desc="Extracting frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % interval == 0:
            timestamp = frame_num / fps
            result = detect_board(frame, min_area=args.min_area)
            if result:
                crop, gray_crop = result
                
                # Apply enhancement if requested
                if args.enhance:
                    crop = enhance_image(crop)
                    gray_crop = enhance_image(gray_crop)
                
                gray_small = cv2.resize(gray_crop, (300, 200))
                boards.append({'timestamp': timestamp, 'orig': crop, 'gray_small': gray_small})
                pbar.update(1)
        frame_num += 1
    
    cap.release()
    pbar.close()
    
    if not boards:
        print("⚠️ No boards detected. Try adjusting detection parameters.")
        return
    
    print(f"✓ Extracted {len(boards)} candidate board images")

    # Prepare metrics for deduplication
    print("✓ Computing image metrics...")
    for b in boards:
        b['sharpness'] = cv2.Laplacian(b['gray_small'], cv2.CV_64F).var()
        b['phash'] = phash(b['gray_small'])

    # Optional deep embeddings
    if tf_available:
        print("✓ Computing deep embeddings...")
        base = ResNet50(weights='imagenet', include_top=False, pooling='avg') if 'ResNet50' in globals() else MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
        preprocess = resnet_preprocess if 'ResNet50' in globals() else mobilenet_preprocess
        size_model = (224,224)
        def embed(img):
            x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x = cv2.resize(x, size_model)
            x = np.expand_dims(x,0)
            x = preprocess(x)
            return base.predict(x, verbose=0).flatten()
        for b in tqdm(boards, desc="Computing embeddings"):
            b['deep'] = embed(b['orig'])
        deep_thresh = 0.85
    else:
        deep_thresh = None
        print("⚠️ TensorFlow not available. Skipping deep embeddings.")

    # Temporal grouping by SSIM & optical flow (OR logic)
    print("✓ Temporal grouping...")
    segments = []
    prev_gray = boards[0]['gray_small']
    for b in boards:
        curr_gray = b['gray_small']
        if not segments:
            segments.append([])
        score = ssim(prev_gray, curr_gray)
        flow_val = mean_flow(prev_gray, curr_gray)
        if score < 0.90 or flow_val > 2.0:
            segments.append([])
        segments[-1].append(b)
        prev_gray = curr_gray
    pre_candidates = [seg[0] for seg in segments if seg]
    
    print(f"✓ Identified {len(segments)} content segments")

    # pHash clustering
    print("✓ Clustering similar boards...")
    clusters = []
    for b in pre_candidates:
        placed = False
        for c in clusters:
            dist = np.count_nonzero(c[0]['phash'] != b['phash'])
            if dist <= phash_thresh:
                c.append(b) ; placed = True; break
        if not placed:
            clusters.append([b])
    
    print(f"✓ Clustered into {len(clusters)} unique board contents")

    # SSIM + deep sub-clustering and select best per sub-cluster
    print("✓ Selecting best quality board for each content...")
    final = []
    for cluster in clusters:
        subclusters = []
        for b in cluster:
            added = False
            for sc in subclusters:
                rep = sc[0]
                cos_ok = True
                if tf_available:
                    cos = np.dot(rep['deep'], b['deep']) / (np.linalg.norm(rep['deep'])*np.linalg.norm(b['deep'])+1e-10)
                    cos_ok = cos >= deep_thresh
                ssim_ok = ssim(rep['gray_small'], b['gray_small']) >= ssim_thresh
                if ssim_ok and cos_ok:
                    sc.append(b) ; added = True; break
            if not added:
                subclusters.append([b])
        for sc in subclusters:
            best = max(sc, key=lambda x: x['sharpness'])
            final.append(best)
    
    print(f"✓ Final selection: {len(final)} high-quality boards")

    # Save results and metadata with improved timestamp-based filenames
    print("✓ Saving board images and metadata...")
    meta = []
    for b in tqdm(sorted(final, key=lambda x: x['timestamp']), desc="Saving images"):
        # compute timestamp components
        total_ms = int(b['timestamp'] * 1000)
        hrs = total_ms // 3600000
        mins = (total_ms % 3600000) // 60000
        secs = (total_ms % 60000) // 1000
        ms = total_ms % 1000
        fname = f"board_{hrs:02d}_{mins:02d}_{secs:02d}_{ms:03d}.jpg"
        path = os.path.join(out_dir, fname)
        
        # Save with configured quality
        cv2.imwrite(path, b['orig'], [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        
        # Format timestamp as hh_mm_ss string instead of seconds
        formatted_timestamp = f"{hrs:02d}_{mins:02d}_{secs:02d}"
        
        # Store only the relative path in the JSON, not the full absolute path
        # This ensures better compatibility when boards.json is used by other tools
        rel_path = os.path.basename(path)
        meta.append({'timestamp': formatted_timestamp, 'path': rel_path})

    with open(os.path.join(out_dir, 'boards.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Successfully extracted and deduplicated {len(final)} boards")
    print(f"✅ Metadata saved to {os.path.join(out_dir, 'boards.json')}")
    
    # Create pre-optimized OCR images if requested
    if args.preoptimize_ocr:
        print("✓ Creating pre-optimized OCR images...")
        ocr_data = []
        
        for b in tqdm(sorted(final, key=lambda x: x['timestamp']), desc="Optimizing for OCR"):
            # Get timestamp components again
            total_ms = int(b['timestamp'] * 1000)
            hrs = total_ms // 3600000
            mins = (total_ms % 3600000) // 60000
            secs = (total_ms % 60000) // 1000
            ms = total_ms % 1000
            
            # Format timestamp and filename
            formatted_timestamp = f"{hrs:02d}_{mins:02d}_{secs:02d}"
            fname = f"board_{hrs:02d}_{mins:02d}_{secs:02d}_{ms:03d}.jpg"
            rel_path = os.path.basename(fname)
            
            # Create OCR-optimized image
            ocr_img = optimize_for_ocr(b['orig'])
            
            # Add to OCR data list
            ocr_data.append({
                'timestamp': formatted_timestamp,
                'path': rel_path,
                'ocr_img': ocr_img
            })
        
        # Save OCR data to pickle file
        pickle_path = os.path.join(out_dir, 'boards_ocr_data.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(ocr_data, f)
        
        print(f"✅ Pre-optimized OCR data saved to {pickle_path}")
        print(f"   This file will be used by LLM_OCR.py for better OCR results")

if __name__ == '__main__':
    main()