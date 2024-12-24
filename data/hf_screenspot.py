import os
import json
from pathlib import Path
import argparse

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
    
    # Process each metadata file
    for meta_file in metadata_dir.glob("*.json"):
        with open(meta_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Create formatted metadata
        formatted_data = {
            "img_url": meta_file.stem + ".png",  # Assuming images are PNG
            "img_size": data.get("image_size", [1920, 1080]),  # Default size if not provided
            "element": []
        }
        
        # Process each element/annotation
        for elem in data.get("annotations", []):
            # Convert bbox from [x1, y1, x2, y2] to normalized [x1/w, y1/h, x2/w, y2/h]
            bbox = elem.get("bbox", [0, 0, 0, 0])
            w, h = formatted_data["img_size"]
            normalized_bbox = [
                bbox[0]/w,
                bbox[1]/h,
                bbox[2]/w,
                bbox[3]/h
            ]
            
            # Calculate center point
            point = [
                (normalized_bbox[0] + normalized_bbox[2])/2,
                (normalized_bbox[1] + normalized_bbox[3])/2
            ]
            
            element_data = {
                "instruction": elem.get("text", ""),
                "bbox": normalized_bbox,
                "data_type": "text",  # Default to text type
                "point": point
            }
            
            formatted_data["element"].append(element_data)
        
        # Add element size
        formatted_data["element_size"] = len(formatted_data["element"])
        
        # Update original metadata file
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Process ScreenSpot dataset metadata')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to ScreenSpot dataset directory')
    
    args = parser.parse_args()
    process_screenspot(args.data_dir)

if __name__ == "__main__":
    main()
