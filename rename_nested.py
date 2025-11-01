"""
Rename images in nested directories to proper format
"""

import os
from pathlib import Path
import random

# Set random seed for reproducibility
random.seed(42)

def rename_nested_images():
    """Rename all images in nested directories."""
    
    base_dir = Path('data/dermnet_images')
    directories_to_rename = [
        'cellulitis', 
        'cold_sores', 
        'eczema', 
        'fungal_infection', 
        'heat_rash'
    ]
    
    total_renamed = 0
    
    for dir_name in directories_to_rename:
        directory_path = base_dir / dir_name
        
        if not directory_path.exists():
            print(f'{dir_name}: Directory not found')
            continue
        
        print(f'Processing {dir_name}...')
        
        # Collect all images
        all_images = []
        for img_file in directory_path.rglob('*'):
            if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                all_images.append(img_file)
        
        # Rename images
        for i, img_file in enumerate(all_images, 1):
            # Get condition name from directory (replace space with underscore)
            condition_name = dir_name.replace(' ', '_')
            
            # Get subdirectory name for additional context
            relative_path = img_file.relative_to(directory_path)
            subdir = relative_path.parent
            
            # Create new filename
            new_name = f"{condition_name}_{i:03d}.jpg"
            new_path = directory_path / new_name
            
            try:
                # Rename the file
                img_file.rename(new_path)
                total_renamed += 1
                if i <= 3 or (i + 1) % 10 == 0:
                    print(f'  Renamed: {img_file.name} -> {new_name}')
            except Exception as e:
                print(f'  Error renaming {img_file.name}: {e}')
        
        print(f'  Total: {len(all_images)} images renamed')
        print()
    
    print(f'Total images renamed: {total_renamed}')
    print('All images are now ready for training!')

if __name__ == "__main__":
    rename_nested_images()

