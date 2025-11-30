# !pip install "autogluon.tabular==1.0.0"

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.multimodal import MultiModalPredictor

# 1. Load Sentiment140 CSV
csv_path = "training.1600000.processed.noemoticon.csv"  # change path
cols = ['target', 'id', 'date', 'flag', 'user', 'text']
df = pd.read_csv(csv_path, encoding='latin-1', names=cols)

# 2. Create numeric sentiment label: -1, 0, 1
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

# 3. Keep only tweet text as feature (you can add more later if you want)
data = df[['text', label_col]]

# Optional: subsample for speed while experimenting
data = data.sample(200_000, random_state=0)

def clean_tweet(t):
    t = str(t)
    t = t.lower()
    t = re.sub(r'http\S+', ' URL ', t)        # remove URLs
    t = re.sub(r'@\w+', ' USER ', t)         # remove @handles
    t = re.sub(r'#', ' ', t)                 # drop hash sign
    t = re.sub(r'\s+', ' ', t).strip()
    return t

data['text'] = data['text'].apply(clean_tweet)

# 4. Train/valid split
train_df, test_df = train_test_split(
    data,
    test_size=0.2,
    random_state=0,
    stratify=data[label_col]
)

train_data = TabularDataset(train_df)
test_data = TabularDataset(test_df)

# 5. Train AutoGluon model (includes Random Forest)
predictor = TabularPredictor(
    label=label_col,
    eval_metric='accuracy',
    path='AutogluonModels/sentimentAnalysis'
).fit(
    train_data,
    presets='best_quality',
    time_limit=7200,
    hyperparameters={
        'RF': {},        # Random Forest
        'GBM': {},       # LightGBM
        'CAT': {},       # CatBoost
        'NN_TORCH': {},  # Neural net
    }
)

# 6. Evaluate
print("Test metrics:")
print(predictor.evaluate(test_data))

# 7. Leaderboard (see how RF compares)
leaderboard = predictor.leaderboard(test_data, silent=False)
)

print("\n=== Transformer (MultiModal) model performance ===")
mm_metrics = mm_predictor.evaluate(test_df)
print(mm_metrics)