"""
Final summary
"""

import pandas as pd

print('FINAL SYSTEM SUMMARY')
print('='*70)
df = pd.read_csv('data/final_expanded_dataset.csv')
print(f'Total images: {len(df)}')
print(f'Total classes: {df["dx"].nunique()}')
print()
print('Class distribution:')
print('='*70)
for condition, count in df['dx'].value_counts().items():
    print(f'{condition:20s}: {count:4d} images')

print()
print('READY TO TRAIN!')
print('='*70)
print('Your system now supports 15 dermatology conditions!')
print('Run: python training/train_simple.py')

