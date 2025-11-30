import pandas as pd
import re
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.multimodal import MultiModalPredictor

# 1. Load Data
csv_path = "ExtractedTweets.csv"
df = pd.read_csv(csv_path)

# 2. Define Label
label_col = 'Party'
print(f"Unique classes: {df[label_col].unique()}")

# 3. Clean Text
def clean_tweet(t):
    t = str(t)
    t = t.lower()
    t = re.sub(r'http\S+', ' URL ', t)        # remove URLs
    t = re.sub(r'@\w+', ' USER ', t)         # remove @handles
    t = re.sub(r'#', ' ', t)                 # drop hash sign
    t = re.sub(r'\s+', ' ', t).strip()
    return t

df['Tweet'] = df['Tweet'].apply(clean_tweet)

# Drop 'Handle' to ensure we predict based on text content only, not the person
df = df[['Tweet', label_col]]

# 4. Train/Test Split
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df[label_col]
)

train_data = TabularDataset(train_df)
test_data = TabularDataset(test_df)

# 5. Train Tabular Model (RF, GBM, etc.)
print("\n=== Training Tabular Model ===")
predictor = TabularPredictor(
    label=label_col,
    eval_metric='accuracy',
    path='AutogluonModels/politicalAffiliation_tabular'
).fit(
    train_data,
    presets='best_quality',
    time_limit=3600, # 1 hour limit
)

# Evaluate Tabular
print("Tabular Test Metrics:")
print(predictor.evaluate(test_data))

# Clean up tabular models (keep only best)
predictor.delete_models(models_to_keep='best', dry_run=False)


# 6. Train MultiModal Model (Transformer)
print("\n=== Training MultiModal Model ===")
mm_predictor = MultiModalPredictor(
    label=label_col,
    # problem_type='multiclass', # Removed to let AutoGluon infer automatically (Binary in this case)
    eval_metric='accuracy',
    path='AutogluonModels/politicalAffiliation_transformer'
)

mm_predictor.fit(
    train_df,
    time_limit=3600, # 1 hour limit
    presets='best_quality'
)

# Evaluate MultiModal
print("MultiModal Test Metrics:")
print(mm_predictor.evaluate(test_df))
