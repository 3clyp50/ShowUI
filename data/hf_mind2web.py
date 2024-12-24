import os
import json
from pathlib import Path
import argparse

def process_showui_desktop(data_dir):
    """
    Process ShowUI-desktop dataset to update metadata in the required format.
    
    Args:
        data_dir: Path to the ShowUI-desktop dataset directory containing images and metadata
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
            "img_size": data.get("viewport_size", [1920, 1080]),  # Default size if not provided
            "element": []
        }
        
        # Process each UI element
        for elem in data.get("elements", []):
            # Get bounding box coordinates
            bbox = elem.get("bbox", {"x": 0, "y": 0, "width": 0, "height": 0})
            w, h = formatted_data["img_size"]
            
            # Convert to normalized coordinates [x1/w, y1/h, x2/w, y2/h]
            x1 = bbox["x"] / w
            y1 = bbox["y"] / h
            x2 = (bbox["x"] + bbox["width"]) / w
            y2 = (bbox["y"] + bbox["height"]) / h
            normalized_bbox = [x1, y1, x2, y2]
            
            # Calculate center point
            point = [
                (x1 + x2) / 2,
                (y1 + y2) / 2
            ]
            
            element_data = {
                "instruction": elem.get("text", ""),
                "bbox": normalized_bbox,
                "data_type": elem.get("tag_name", "text").lower(),  # Use HTML tag name as type
                "point": point
            }
            
            formatted_data["element"].append(element_data)
        
        # Add element size
        formatted_data["element_size"] = len(formatted_data["element"])
        
        # Update original metadata file
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Process ShowUI-desktop dataset metadata')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to ShowUI-desktop dataset directory')
    
    args = parser.parse_args()
    process_showui_desktop(args.data_dir)

if __name__ == "__main__":
    main()
