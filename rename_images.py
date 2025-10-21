"""
Image Renamer for Dermatology Dataset
Renames downloaded images to proper training format
"""

import os
from pathlib import Path
import shutil

def rename_acne_images():
    """Rename acne images to proper format."""
    
    # Source directory
    source_dir = Path("data/dermnet_images/acne")
    
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
        new_name = f"acne_{i:03d}.jpg"
        new_path = source_dir / new_name
        
        try:
            # Rename the file
            img_file.rename(new_path)
            print(f"Renamed: {img_file.name} -> {new_name}")
            renamed_count += 1
            
        except Exception as e:
            print(f"Error renaming {img_file.name}: {e}")
    
    print(f"\nRenamed {renamed_count} images successfully!")
    print(f"Images are now ready for training with format: acne_001.jpg, acne_002.jpg, etc.")

def rename_all_conditions():
    """Rename images in all condition directories."""
    
    base_dir = Path("data/dermnet_images")
    
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return
    
    # Find all condition directories
    condition_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(condition_dirs)} condition directories:")
    for dir in condition_dirs:
        print(f"  - {dir.name}")
    
    total_renamed = 0
    
    for condition_dir in condition_dirs:
        condition_name = condition_dir.name
        print(f"\nProcessing {condition_name}...")
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(list(condition_dir.glob(ext)))
        
        print(f"Found {len(image_files)} images")
        
        # Rename files
        renamed_count = 0
        
        for i, img_file in enumerate(image_files, 1):
            # Create new filename
            new_name = f"{condition_name}_{i:03d}.jpg"
            new_path = condition_dir / new_name
            
            try:
                # Rename the file
                img_file.rename(new_path)
                print(f"  Renamed: {img_file.name} -> {new_name}")
                renamed_count += 1
                
            except Exception as e:
                print(f"  Error renaming {img_file.name}: {e}")
        
        print(f"Renamed {renamed_count} images for {condition_name}")
        total_renamed += renamed_count
    
    print(f"\nTotal renamed: {total_renamed} images")
    print("All images are now ready for training!")

def create_metadata_for_renamed():
    """Create metadata CSV for renamed images."""
    
    import pandas as pd
    
    base_dir = Path("data/dermnet_images")
    
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}")
        return
    
    rows = []
    class_id = 8  # Start from class 8 (after existing 7 + non_skin)
    
    # Find all condition directories
    condition_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    for condition_dir in condition_dirs:
        condition_name = condition_dir.name
        
        # Get all renamed image files
        image_files = list(condition_dir.glob("*.jpg"))
        
        print(f"Found {len(image_files)} images for {condition_name}")
        
        for img_file in image_files:
            rows.append({
                "image": img_file.stem,  # filename without extension
                "label_id": class_id,
                "dx": condition_name,
                "age": "",
                "sex": "",
                "localization": "unknown"
            })
        
        class_id += 1
    
    if rows:
        df = pd.DataFrame(rows)
        output_path = "data/dermnet_images_metadata.csv"
        df.to_csv(output_path, index=False)
        
        print(f"\nGenerated metadata for {len(rows)} images")
        print(f"Saved to: {output_path}")
        
        # Show class distribution
        print("\nClass distribution:")
        class_counts = df['dx'].value_counts()
        for condition, count in class_counts.items():
            print(f"  {condition}: {count} images")
    else:
        print("No images found!")

def main():
    """Main function with options."""
    
    print("Dermatology Image Renamer")
    print("="*50)
    print("Choose an option:")
    print("1. Rename only acne images")
    print("2. Rename all condition images")
    print("3. Rename all + create metadata")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        rename_acne_images()
    elif choice == "2":
        rename_all_conditions()
    elif choice == "3":
        rename_all_conditions()
        create_metadata_for_renamed()
    else:
        print("Invalid choice. Running option 1 (acne only)...")
        rename_acne_images()

if __name__ == "__main__":
    main()
