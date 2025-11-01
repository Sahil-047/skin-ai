"""
Analyze dataset growth
"""

import pandas as pd

# Load all datasets
original = pd.read_csv('data/metadata_final.csv')
with_nonskin = pd.read_csv('data/metadata_final_with_nonskin.csv')
with_acne = pd.read_csv('data/expanded_with_acne.csv')
with_athletes_foot = pd.read_csv('data/expanded_with_athletes_foot.csv')

print('DATASET GROWTH ANALYSIS')
print('='*60)
print(f'Original dataset (ISIC): {len(original)} images')
print(f'After adding non_skin: {len(with_nonskin)} images (+{len(with_nonskin)-len(original)} images)')
print(f'After adding acne: {len(with_acne)} images (+{len(with_acne)-len(with_nonskin)} images)')
print(f'After adding athletes foot: {len(with_athletes_foot)} images (+{len(with_athletes_foot)-len(with_acne)} images)')
print()
print('TOTAL GROWTH:')
print(f'  Original: {len(original)} images')
print(f'  Final: {len(with_athletes_foot)} images')
print(f'  Total added: {len(with_athletes_foot)-len(original)} new images')
print()
print('BREAKDOWN BY CLASS:')
print('='*60)
print('Class distribution in final dataset:')
class_counts = with_athletes_foot['dx'].value_counts()
for condition, count in class_counts.items():
    print(f'  {condition}: {count} images')

print()
print('NEW DATA YOU ADDED:')
print('='*60)
print('1. Non-skin images:')
nonskin_data = with_nonskin[with_nonskin['dx'] == 'non_skin']
print(f'   Count: {len(nonskin_data)} images')
print(f'   Age range: {nonskin_data["age"].min()}-{nonskin_data["age"].max()}')
print(f'   Sex: {nonskin_data["sex"].value_counts().to_dict()}')

print()
print('2. Acne images:')
acne_data = with_athletes_foot[with_athletes_foot['dx'] == 'acne']
print(f'   Count: {len(acne_data)} images')
print(f'   Age range: {acne_data["age"].min()}-{acne_data["age"].max()}')
print(f'   Sex: {acne_data["sex"].value_counts().to_dict()}')
print(f'   Localization: {acne_data["localization"].value_counts().to_dict()}')

print()
print('3. Athlete\'s foot images:')
athletes_foot_data = with_athletes_foot[with_athletes_foot['dx'] == 'athletes_foot']
print(f'   Count: {len(athletes_foot_data)} images')
print(f'   Age range: {athletes_foot_data["age"].min()}-{athletes_foot_data["age"].max()}')
print(f'   Sex: {athletes_foot_data["sex"].value_counts().to_dict()}')
print(f'   Localization: {athletes_foot_data["localization"].value_counts().to_dict()}')

print()
print('SUMMARY:')
print(f'  Total new classes: 3 (non_skin, acne, athletes_foot)')
print(f'  Total new images: {len(with_athletes_foot)-len(original)} images')
print(f'  Percentage increase: {(len(with_athletes_foot)-len(original))/len(original)*100:.1f}%')
print()
print('MODEL UPDATE:')
print(f'  Classes: 7 -> 10 (+3 new classes)')
print(f'  Images: {len(original)} -> {len(with_athletes_foot)} (+{len(with_athletes_foot)-len(original)} images)')

