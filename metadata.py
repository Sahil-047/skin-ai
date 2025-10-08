import pandas as pd

df = pd.read_csv("data\HAM10000_metadata.csv")
ham_labels = pd.read_csv("data\Ham_labels.csv")
isic_labels = pd.read_csv("data\isic_labels.csv")

# Merge using image_id
merged = df.merge(ham_labels, how="left", on="image_id")
merged = merged.merge(isic_labels, how="left", on="image_id")

# Pick the primary 'dx' column
# Prefer dx_x (HAM metadata) if not null, else fallback to dx_y or dx
merged["dx_final"] = merged["dx_x"].combine_first(merged["dx_y"]).combine_first(merged["dx"])

# Drop unnecessary columns
clean = merged.drop(columns=["dx_x", "dx_y", "dx"], errors="ignore")

# Map disease names → numeric labels
label_map = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6
}
clean["label_id"] = clean["dx_final"].map(label_map)

# Drop rows missing labels
clean = clean.dropna(subset=["label_id"])

# Final column order
clean = clean[["image_id", "label_id", "dx_final", "age", "sex", "localization"]]
clean = clean.rename(columns={"image_id": "image", "dx_final": "dx"})

# Save the unified metadata
clean.to_csv("metadata_final.csv", index=False)
print("✅ Final merged dataset saved as metadata_final.csv")
print("Rows:", len(clean))
print(clean.head())