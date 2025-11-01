"""
Count images in data directory
"""

import os
from pathlib import Path

# Check directories for image counts
base_dir = Path('data')
dermnet_dir = base_dir / 'dermnet_images'
non_skin_dir = base_dir / 'non_skin'
images_dir = base_dir / 'images'

print('IMAGE COUNT ANALYSIS')
print('='*60)

# Count images in each directory
counts = {}
total_images = 0

# Check dermnet_images subdirectories
if dermnet_dir.exists():
    for subdir in dermnet_dir.iterdir():
        if subdir.is_dir():
            img_count = len(list(subdir.glob('*.jpg'))) + len(list(subdir.glob('*.jpeg')))
            if img_count > 0:
                counts[subdir.name] = img_count
                total_images += img_count

# Check non_skin directory
if non_skin_dir.exists():
    img_count = len(list(non_skin_dir.glob('*.jpg'))) + len(list(non_skin_dir.glob('*.jpeg')))
    if img_count > 0:
        counts['non_skin'] = img_count
        total_images += img_count

# Count original ISIC images
if images_dir.exists():
    img_count = len(list(images_dir.glob('*.jpg')))
    counts['original_isic'] = img_count

print('Images by directory:')
print('='*60)
for dir_name, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
    print(f'{dir_name}: {count} images')

print()
print('NEW DATA YOU ADDED:')
print('='*60)
new_data_total = counts.get('non_skin', 0) + counts.get('acne', 0) + counts.get("athlete's_foot", 0)
print(f'Total new images added: {new_data_total} images')
print()
print('Breakdown:')
print(f'  non_skin: {counts.get("non_skin", 0)} images')
print(f'  acne: {counts.get("acne", 0)} images')
athletes_foot_count = counts.get("athlete's_foot", 0)
print(f'  athletes_foot: {athletes_foot_count} images')

print()
print('ORIGINAL DATA:')
print('='*60)
print(f'  original_isic: {counts.get("original_isic", 0)} images')

print()
print('TOTAL IMAGES:')
print('='*60)
print(f'  Original: {counts.get("original_isic", 0)} images')
print(f'  New: {new_data_total} images')
print(f'  Total: {counts.get("original_isic", 0) + new_data_total} images')
