import os
import json
from pathlib import Path
import argparse
import time

def process_screenspot(data_dir):
    """
    Process ScreenSpot dataset to update metadata in the required format.
    
    Args:
        data_dir: Path to the ScreenSpot dataset directory containing images and metadata
    """
    # Ensure the paths exist
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    metadata_dir = data_dir / "metadata"
    
    if not all(p.exists() for p in [images_dir, metadata_dir]):
        raise ValueError(f"Required directories not found in {data_dir}")
    
    start_time = time.time()
    print(f"Processing metadata files in {metadata_dir}")
    
    # Initialize counters
    file_count = 0
    success_count = 0
    failed_files = []
    total_elements = 0
    invalid_elements = 0
    for meta_file in metadata_dir.glob("*.json"):
        file_count += 1
        print(f"Processing file {file_count}: {meta_file.name}")
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Error: Invalid JSON in file {meta_file.name}: {e}")
                    failed_files.append(f"{meta_file.name} (Invalid JSON)")
                    continue
        except Exception as e:
            print(f"Error: Could not read file {meta_file.name}: {e}")
            failed_files.append(f"{meta_file.name} (Read Error)")
            continue
            
        # Get and validate image size
        try:
            raw_size = data[0].get("image_size", [1920, 1080]) if isinstance(data, list) and len(data) > 0 else [1920, 1080]
            validated_size = [
                abs(raw_size[0]) if len(raw_size) > 0 else 1920,
                abs(raw_size[1]) if len(raw_size) > 1 else 1080
            ]
        except (IndexError, TypeError, ValueError) as e:
            print(f"Error: Invalid image size in file {meta_file.name}: {e}")
            failed_files.append(f"{meta_file.name} (Invalid Image Size)")
            continue
        
        # Create formatted metadata
        formatted_data = {
            "img_url": meta_file.stem + ".png",  # Assuming images are PNG
            "img_size": validated_size,  # Use validated dimensions
            "element": []
        }
        
        # Process each element/annotation
        annotations = data if isinstance(data, list) else data.get("annotations", [])
        for elem in annotations:
            try:
                # Convert bbox from [x1, y1, x2, y2] to normalized [x1/w, y1/h, x2/w, y2/h]
                # Ensure bbox has exactly 4 values, default to [0,0,0,0] if malformed
                raw_bbox = elem.get("bbox", [0, 0, 0, 0])
                bbox = raw_bbox[:4] if len(raw_bbox) >= 4 else [0, 0, 0, 0]
                w, h = formatted_data["img_size"]
                # Avoid division by zero
                normalized_bbox = [
                    bbox[0]/w if w != 0 else 0,
                    bbox[1]/h if h != 0 else 0,
                    bbox[2]/w if w != 0 else 0,
                    bbox[3]/h if h != 0 else 0
                ]
                
                # Calculate center point
                point = [
                    (normalized_bbox[0] + normalized_bbox[2])/2,
                    (normalized_bbox[1] + normalized_bbox[3])/2
                ]
            except (IndexError, TypeError, ValueError) as e:
                print(f"Warning: Invalid bbox in file {meta_file.name}, element {len(formatted_data['element']) + 1}: {e}")
                normalized_bbox = [0, 0, 0, 0]
                point = [0, 0]
                invalid_elements += 1
            
            element_data = {
                "instruction": elem.get("text", ""),
                "bbox": normalized_bbox,
                "data_type": "text",  # Default to text type
                "point": point
            }
            
            formatted_data["element"].append(element_data)
            total_elements += 1
        
        # Add element size
        formatted_data["element_size"] = len(formatted_data["element"])
        
        # Update original metadata file
        try:
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(formatted_data, f, indent=2)
            success_count += 1
        except Exception as e:
            print(f"Error: Could not write to file {meta_file.name}: {e}")
            failed_files.append(f"{meta_file.name} (Write Error)")
            continue
    
    print(f"\nProcessing complete:")
    print(f"- Total files found: {file_count}")
    print(f"- Successfully processed: {success_count}")
    print(f"- Failed to process: {len(failed_files)}")
    if failed_files:
        print("- Failed files:")
        for file in failed_files:
            print(f"  - {file}")
    print(f"- Output directory: {metadata_dir}")
    print(f"- Total elements processed: {total_elements}")
    if invalid_elements > 0:
        print(f"- Invalid elements encountered: {invalid_elements}")
    
    elapsed_time = time.time() - start_time
    print(f"- Total processing time: {elapsed_time:.2f} seconds")
    if success_count > 0:
        avg_time = elapsed_time / success_count
        print(f"- Average time per file: {avg_time:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description='Process ScreenSpot dataset metadata')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to ScreenSpot dataset directory')
    
    args = parser.parse_args()
    process_screenspot(args.data_dir)

if __name__ == "__main__":
    main()
