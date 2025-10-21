"""
Merge acne metadata with expanded dataset
"""

import pandas as pd

def merge_datasets():
    """Merge acne metadata with existing dataset."""
    
    # Load datasets
    existing = pd.read_csv('data/metadata_final_with_nonskin.csv')
    acne = pd.read_csv('data/acne_metadata.csv')
    
    # Combine datasets
    combined = pd.concat([existing, acne], ignore_index=True)
    
    # Save combined dataset
    combined.to_csv('data/expanded_with_acne.csv', index=False)
    
    print(f'Combined dataset: {len(combined)} images')
    print('\nClass distribution:')
    class_counts = combined['dx'].value_counts()
    for condition, count in class_counts.items():
        print(f'  {condition}: {count} images')
    
    print('\nAcne metadata includes:')
    acne_data = combined[combined['dx'] == 'acne']
    print(f'  Age range: {acne_data["age"].min()}-{acne_data["age"].max()}')
    print(f'  Sex distribution: {acne_data["sex"].value_counts().to_dict()}')
    print(f'  Localization: {acne_data["localization"].value_counts().to_dict()}')
    
    print('\nDataset ready for training!')

if __name__ == "__main__":
    merge_datasets()
