import os
import cv2
import argparse
import json
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Optional deep-feature embeddings
try:
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_preprocess
    tf_available = True
except ImportError:
    tf_available = False

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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boards = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            ar = w / float(h)
            if area > 150000 and 1.0 < ar < 3.5:
                boards.append((x,y,w,h))
    if not boards:
        return None
    x,y,w,h = max(boards, key=lambda b: b[2]*b[3])
    return frame[y:y+h, x:x+w], gray[y:y+h, x:x+w]

# Main pipeline
def main():
    parser = argparse.ArgumentParser(description="Extract and deduplicate blackboard images from lecture video.")
    parser.add_argument("-i", "--input", required=True, help="Path to input .mp4 video file")
    parser.add_argument("-o", "--output", required=True, help="Directory to save final images and metadata")
    args = parser.parse_args()

    video_path = args.input
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    # Step 1: Extract board crops with timestamps
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = int(fps * 3)  # 1 frame per 3 seconds
    boards = []  # list of dicts: {'timestamp': , 'orig': img, 'gray_small': }

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % interval == 0:
            timestamp = frame_num / fps
            result = detect_board(frame)
            if result:
                crop, gray_crop = result
                gray_small = cv2.resize(gray_crop, (300,200))
                boards.append({'timestamp': timestamp, 'orig': crop, 'gray_small': gray_small})
        frame_num += 1
    cap.release()

    # Prepare metrics for deduplication
    for b in boards:
        b['sharpness'] = cv2.Laplacian(b['gray_small'], cv2.CV_64F).var()
        b['phash'] = phash(b['gray_small'])

    # Optional deep embeddings
    if tf_available:
        base = ResNet50(weights='imagenet', include_top=False, pooling='avg') if 'ResNet50' in globals() else MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
        preprocess = resnet_preprocess if 'ResNet50' in globals() else mobilenet_preprocess
        size_model = (224,224)
        def embed(img):
            x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x = cv2.resize(x, size_model)
            x = np.expand_dims(x,0)
            x = preprocess(x)
            return base.predict(x).flatten()
        for b in boards:
            b['deep'] = embed(b['orig'])

    # Deduplication parameters
    phash_thresh = 20
    ssim_thresh = 0.80
    deep_thresh = 0.85 if tf_available else None

    # Temporal grouping by SSIM & optical flow (OR logic)
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

    # pHash clustering
    clusters = []
    for b in pre_candidates:
        placed = False
        for c in clusters:
            dist = np.count_nonzero(c[0]['phash'] != b['phash'])
            if dist <= phash_thresh:
                c.append(b) ; placed = True; break
        if not placed:
            clusters.append([b])

    # SSIM + deep sub-clustering and select best per sub-cluster
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

    # Save results and metadata with improved timestamp-based filenames
    meta = []
    for b in sorted(final, key=lambda x: x['timestamp']):
        # compute timestamp components
        total_ms = int(b['timestamp'] * 1000)
        hrs = total_ms // 3600000
        mins = (total_ms % 3600000) // 60000
        secs = (total_ms % 60000) // 1000
        ms = total_ms % 1000
        fname = f"board_{hrs:02d}_{mins:02d}_{secs:02d}_{ms:03d}.jpg"
        path = os.path.join(out_dir, fname)
        cv2.imwrite(path, b['orig'])
        # Format timestamp as hh_mm_ss string instead of seconds
        formatted_timestamp = f"{hrs:02d}_{mins:02d}_{secs:02d}"
        meta.append({'timestamp': formatted_timestamp, 'path': path})

    with open(os.path.join(out_dir, 'boards.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Extracted and deduplicated {len(final)} boards. Metadata saved to boards.json")

if __name__ == '__main__':
    main()
