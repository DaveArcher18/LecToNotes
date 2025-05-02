#!/usr/bin/env python3

import os
import argparse
import cv2
import matplotlib.pyplot as plt
from LLM_OCR import (
    optimize_image_for_ocr,
    encode_image_to_data_uri,
    multi_stage_ocr_process
)

def main():
    """Test the OCR functionality on a single image or directory of images."""
    parser = argparse.ArgumentParser(
        description="Test the OCR functionality on images."
    )
    parser.add_argument("input_path", 
                      help="Path to an image or directory of images")
    parser.add_argument("--show-preprocessing", action="store_true",
                      help="Display preprocessing steps for each image")
    parser.add_argument("--save-results", action="store_true",
                      help="Save OCR results to a text file alongside each image")
    args = parser.parse_args()
    
    # Handle directory or single file
    if os.path.isdir(args.input_path):
        # Process all images in directory
        for filename in os.listdir(args.input_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                process_image(os.path.join(args.input_path, filename), args)
    else:
        # Process single image
        process_image(args.input_path, args)

def process_image(image_path, args):
    """Process a single image with OCR."""
    print(f"\nProcessing {os.path.basename(image_path)}...")
    
    # Show preprocessing if requested
    if args.show_preprocessing:
        img = cv2.imread(image_path)
        img_proc = optimize_image_for_ocr(image_path)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB))
        plt.title("OCR-Optimized Image")
        plt.axis("off")
        
        plt.tight_layout()
        plt.show()
    
    # Run OCR
    print("Extracting text using multi-stage OCR process...")
    latex_text = multi_stage_ocr_process(image_path)
    
    # Display results
    print("\n----- OCR Result -----")
    print(latex_text)
    print("-----------------------")
    
    # Save results if requested
    if args.save_results:
        base_path = os.path.splitext(image_path)[0]
        output_file = f"{base_path}_ocr_result.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(latex_text)
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main() 