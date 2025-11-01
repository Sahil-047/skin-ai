"""
Merge all metadata and update system
"""

import pandas as pd

print('MERGING ALL DATA')
print('='*60)

# Load existing dataset
existing = pd.read_csv('data/expanded_with_athletes_foot.csv')
new_conditions = pd.read_csv('data/new_conditions_metadata.csv')

print(f'Existing dataset: {len(existing)} images')
print(f'New conditions: {len(new_conditions)} images')

# Merge datasets
combined = pd.concat([existing, new_conditions], ignore_index=True)

print(f'Combined dataset: {len(combined)} images')
print()
print('Class distribution:')
print('='*60)
class_counts = combined['dx'].value_counts()
for condition, count in class_counts.items():
    print(f'  {condition}: {count} images')

# Save combined dataset
combined.to_csv('data/final_expanded_dataset.csv', index=False)

print()
print('='*60)
print(f'Final dataset saved to: data/final_expanded_dataset.csv')
print(f'Total images: {len(combined)} images')
print(f'Total classes: {len(class_counts)} classes')
print()
print('NEW CLASSES ADDED:')
print('='*60)
new_classes = ['cellulitis', 'cold_sores', 'eczema', 'fungal_infection', 'heat_rash']
for condition in new_classes:
    count = len(combined[combined['dx'] == condition])
    print(f'  {condition}: {count} images')

print()
print('READY FOR TRAINING!')
print('='*60)
print('Next steps:')
print('1. Update training script to use 15 classes')
print('2. Update backend to support 15 classes')
print('3. Update dataset.py to load new images')
print('4. Train the model')

