import pandas as pd
import nlpaug.augmenter.word as naw
import nltk
import os
import shutil

# ✅ Fix NLTK resource issues
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("omw-1.4")

try:
    nltk.download("averaged_perceptron_tagger_eng")
except:
    try:
        tagger_path = nltk.data.find("taggers/averaged_perceptron_tagger")
        eng_path = os.path.join(os.path.dirname(tagger_path), "averaged_perceptron_tagger_eng")
        if not os.path.exists(eng_path):
            shutil.copytree(tagger_path, eng_path)
    except LookupError:
        pass

# ✅ CONFIG
AUG_PER_ROW = 3  # 🔥 Number of new variations per row
INPUT_FILE = "darkweb_data.csv"  # Your base dataset
OUTPUT_FILE = "darkweb_data_augmented.csv"

# ✅ Load dataset
df = pd.read_csv(INPUT_FILE)

# ✅ Synonym-based augmenter
augmenter = naw.SynonymAug(aug_src="wordnet")

augmented_rows = []

print(f"🚀 Starting augmentation for {len(df)} rows...")

for index, row in df.iterrows():
    original_text = row["text"]
    label = row["label"]
    row_type = row["type"] if "type" in df.columns else None

    try:
        augmented_texts = augmenter.augment(original_text, n=AUG_PER_ROW)
    except Exception as e:
        print(f"⚠️ Skipping '{original_text}' due to error: {e}")
        continue

    for aug_text in augmented_texts:
        new_row = {"text": aug_text, "label": label}
        if row_type:
            new_row["type"] = row_type
        augmented_rows.append(new_row)

# ✅ Combine datasets
df_augmented = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)

# ✅ Save output
df_augmented.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Augmentation complete! Final dataset has {len(df_augmented)} rows.")
print(f"📂 Saved to: {OUTPUT_FILE}")
