"""
Count images in nested directory structure
"""

from pathlib import Path

# Check all the directories
base_dir = Path('data/dermnet_images')
directories_to_check = ['cellulitis', 'cold_sores', 'eczema', 'fungal_infection', 'heat_rash']

print('NESTED IMAGE COUNT ANALYSIS')
print('='*70)

total_new_images = 0

for directory_name in directories_to_check:
    directory_path = base_dir / directory_name
    
    if not directory_path.exists():
        print(f'{directory_name}: Directory not found')
        continue
    
    # Count all images recursively
    image_count = 0
    subdir_counts = {}
    
    # Get all subdirectories
    for subdir in directory_path.rglob('*'):
        if subdir.is_dir():
            # Count images in this subdirectory
            count = len(list(subdir.glob('*.jpg'))) + len(list(subdir.glob('*.jpeg'))) + len(list(subdir.glob('*.png')))
            if count > 0:
                subdir_counts[subdir.relative_to(directory_path)] = count
                image_count += count
    
    print(f'{directory_name}:')
    print(f'  Total images: {image_count}')
    
    if subdir_counts:
        print('  By subdirectory:')
        for subdir, count in sorted(subdir_counts.items()):
            print(f'    {subdir}: {count} images')
    
    total_new_images += image_count
    print()

print('='*70)
print(f'Total new images in nested directories: {total_new_images} images')
print()
print('Summary:')
for directory_name in directories_to_check:
    directory_path = base_dir / directory_name
    if directory_path.exists():
        count = sum(
            len(list((base_dir / directory_name).rglob(f'*.{ext}')))
            for ext in ['jpg', 'jpeg', 'png']
        )
        print(f'  {directory_name}: {count} images')

