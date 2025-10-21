"""
Rename Athlete's Foot Images
Renames athlete's foot images to proper training format
"""

import os
from pathlib import Path

def rename_athletes_foot_images():
    """Rename athlete's foot images to proper format."""
    
    # Source directory
    source_dir = Path("data/dermnet_images/athlete's_foot")
    
    if not source_dir.exists():
        print(f"Directory not found: {source_dir}")
        return
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(list(source_dir.glob(ext)))
    
    print(f"Found {len(image_files)} images in {source_dir}")
    
    # Rename files
    renamed_count = 0
    
    for i, img_file in enumerate(image_files, 1):
        # Create new filename
        new_name = f"athletes_foot_{i:03d}.jpg"
        new_path = source_dir / new_name
        
        try:
            # Rename the file
            img_file.rename(new_path)
            print(f"Renamed: {img_file.name} -> {new_name}")
            renamed_count += 1
            
        except Exception as e:
            print(f"Error renaming {img_file.name}: {e}")
    
    print(f"\nRenamed {renamed_count} images successfully!")
    print(f"Images are now ready for training with format: athletes_foot_001.jpg, athletes_foot_002.jpg, etc.")

if __name__ == "__main__":
    rename_athletes_foot_images()
