# Load dataset, map labels, sample 20000 each, and save a new CSV at /mnt/data/reviews_40000.csv
import pandas as pd


input_path = r"C:\Users\HP\Downloads\fake reviews dataset.csv"
output_path = r"C:\documents\AI_PROJECT\New folder\reviews_40000.csv"
# Read with robust options
df = pd.read_csv(input_path, encoding='utf-8', on_bad_lines='skip')

# Standardize column names if needed
# Known columns: category, rating, label, text_
cols = df.columns.tolist()

# Rename text_ -> review_text for clarity
if "text_" in df.columns:
    df = df.rename(columns={"text_": "review_text"})

# Map labels: CG->fake, OR->real
df["label_standard"] = df["label"].map({"CG": "fake", "OR": "real"}).fillna(df["label"])

# Count available per class
counts = df["label_standard"].value_counts().to_dict()

# Determine how many to sample
target_each = 20000

# Check available counts
available_fake = int(counts.get("fake", 0))
available_real = int(counts.get("real", 0))

# If not enough for either class, adjust to maximum available
if available_fake < target_each or available_real < target_each:
    # Use min available for both to maintain balance if insufficient
    max_each = min(available_fake, available_real)
    sample_each = max_each
else:
    sample_each = target_each

# Sample
fake_df = df[df["label_standard"] == "fake"].sample(n=sample_each, random_state=42)
real_df = df[df["label_standard"] == "real"].sample(n=sample_each, random_state=42)

balanced_df = pd.concat([fake_df, real_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# For consistency, create final columns: review_text, label (fake/real), category, rating
final_cols = []
if "review_text" in balanced_df.columns:
    final_cols.append("review_text")
elif "text" in balanced_df.columns:
    final_cols.append("text")
if "label_standard" in balanced_df.columns:
    final_cols.append("label_standard")
elif "label" in balanced_df.columns:
    final_cols.append("label")
if "category" in balanced_df.columns:
    final_cols.append("category")
if "rating" in balanced_df.columns:
    final_cols.append("rating")

# Ensure columns exist
final_cols = [c for c in final_cols if c in balanced_df.columns]

final_df = balanced_df[final_cols].rename(columns={"label_standard": "label"})

# Save output
final_df.to_csv(output_path, index=False)

# Report back counts and file path
result = {
    "output_path": output_path,
    "rows": len(final_df),
    "counts": final_df["label"].value_counts().to_dict(),
    "columns": final_df.columns.tolist()
}

result