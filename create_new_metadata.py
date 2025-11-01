"""
Create metadata for new conditions
"""

import pandas as pd
import random
from pathlib import Path

# Set random seed
random.seed(42)

conditions = [
    ('cellulitis', 10),
    ('cold_sores', 11),
    ('eczema', 12),
    ('fungal_infection', 13),
    ('heat_rash', 14)
]

print('Creating metadata for new conditions...')
print('='*60)

all_metadata = []

for condition_name, class_id in conditions:
    directory_path = Path(f'data/dermnet_images/{condition_name}')
    
    if not directory_path.exists():
        print(f'{condition_name}: Directory not found')
        continue
    
    # Get all images
    image_files = list(directory_path.glob('*.jpg'))
    
    print(f'{condition_name}: {len(image_files)} images')
    
    # Generate random metadata
    ages = [random.randint(18, 65) for _ in range(len(image_files))]
    sexes = [random.choice(['male', 'female']) for _ in range(len(image_files))]
    
    # Common localizations for each condition
    localizations_map = {
        'cellulitis': ['legs', 'face', 'hands'],
        'cold_sores': ['lips', 'mouth'],
        'eczema': ['hand', 'legs', 'back', 'chest', 'scalp'],
        'fungal_infection': ['legs', 'hand', 'back', 'chest', 'groin', 'scalp'],
        'heat_rash': ['thighs', 'back', 'chest']
    }
    
    for i, img_file in enumerate(image_files):
        # Randomly assign localization
        localization = random.choice(localizations_map[condition_name])
        
        all_metadata.append({
            'image': img_file.stem,
            'label_id': class_id,
            'dx': condition_name,
            'age': ages[i],
            'sex': sexes[i],
            'localization': localization
        })

# Create DataFrame and save
df = pd.DataFrame(all_metadata)
df.to_csv('data/new_conditions_metadata.csv', index=False)

print('='*60)
print(f'Total images: {len(df)}')
print()
print('Summary by condition:')
print('='*60)
for condition_name, class_id in conditions:
    condition_data = df[df['dx'] == condition_name]
    if len(condition_data) > 0:
        print(f'{condition_name}:')
        print(f'  Count: {len(condition_data)} images')
        print(f'  Age range: {condition_data["age"].min()}-{condition_data["age"].max()}')
        print(f'  Sex: {condition_data["sex"].value_counts().to_dict()}')
        print(f'  Localization: {condition_data["localization"].value_counts().to_dict()}')
        print()

print('Metadata saved to: data/new_conditions_metadata.csv')

