import pandas as pd
import re
import os

# --- 1. Load Data ---
csv_path = r"C:\Users\kevin\Documents\GitHub\open-vRAG\sentiment-analysis\training.1600000.processed.noemoticon.csv"
if not os.path.exists(csv_path):
    print(f"ERROR: File '{csv_path}' not found.")
    import sys
    sys.exit()

print(f"Loading dataset from '{csv_path}'...")
cols = ['target', 'id', 'date', 'flag', 'user', 'text']
df = pd.read_csv(csv_path, encoding='latin-1', names=cols)

# --- 2. Define Label and Clean Data ---
# Create numeric sentiment label: -1, 0, 1
# Original Sentiment140: 0 = negative, 2 = neutral (if present), 4 = positive
label_col = 'sentiment_3class'

mapping_3class = {
    0: -1,   # negative
    2:  0,   # neutral
    4:  1    # positive
}

df[label_col] = df['target'].map(mapping_3class)

# Drop rows with labels not in {0,2,4} (or any NaNs from the mapping)
df = df.dropna(subset=[label_col])

# Keep only tweet text as feature
data = df[['text', label_col]]

def clean_tweet(t):
    t = str(t)
    t = t.lower()
    t = re.sub(r'http\S+', ' URL ', t)        # remove URLs
    t = re.sub(r'@\w+', ' USER ', t)         # remove @handles
    t = re.sub(r'#', ' ', t)                 # drop hash sign
    t = re.sub(r'\s+', ' ', t).strip()
    return t

data['text'] = data['text'].apply(clean_tweet)

# --- 3. Save Processed Data ---
output_path = "C:\\Users\\kevin\\Documents\\GitHub\\open-vRAG\\sentiment-analysis\\processed.csv"
print(f"Saving processed data to '{output_path}'...")
data.to_csv(output_path, index=False)
print("Done.")
