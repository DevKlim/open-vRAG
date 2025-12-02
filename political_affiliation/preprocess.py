import pandas as pd
import re
import os

# --- 1. Load Data ---
csv_path = "C:\\Users\\kevin\\Documents\\GitHub\\open-vRAG\\political_affiliation\\ExtractedTweets.csv"
if not os.path.exists(csv_path):
    print(f"ERROR: File '{csv_path}' not found.")
    import sys
    sys.exit()

print(f"Loading dataset from '{csv_path}'...")
df = pd.read_csv(csv_path)

# --- 2. Define Label and Clean Data ---
label_col = 'Party'
print(f"Unique classes: {df[label_col].unique()}")

def clean_tweet(t):
    t = str(t)
    t = t.lower()
    t = re.sub(r'http\S+', ' URL ', t)        # remove URLs
    t = re.sub(r'@\w+', ' USER ', t)         # remove @handles
    t = re.sub(r'#', ' ', t)                 # drop hash sign
    t = re.sub(r'\s+', ' ', t).strip()
    return t

df['Tweet'] = df['Tweet'].apply(clean_tweet)

# Drop 'Handle' to ensure we predict based on text content only
df = df[['Tweet', label_col]]

# --- 3. Save Processed Data ---
output_path = "C:\\Users\\kevin\\Documents\\GitHub\\open-vRAG\\political_affiliation\\processed.csv"
print(f"Saving processed data to '{output_path}'...")
df.to_csv(output_path, index=False)
print("Done.")
