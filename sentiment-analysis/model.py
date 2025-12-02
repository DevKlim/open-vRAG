# !pip install "autogluon.tabular==1.0.0"

import pandas as pd
import re
import os
import shutil
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import classification_report

# --- 1. Load Data ---
csv_path = "C:\\Users\\kevin\\Documents\\GitHub\\open-vRAG\\sentiment-analysis\\processed.csv"
if not os.path.exists(csv_path):
    print(f"ERROR: File '{csv_path}' not found.")
    print("Please run 'preprocess.py' first.")
    import sys
    sys.exit()

print(f"Loading dataset from '{csv_path}'...")
df = pd.read_csv(csv_path)

# --- 2. Define Label ---
label_col = 'sentiment_3class'
print(f"Unique classes: {df[label_col].unique()}")

# Note: Data is already cleaned by preprocess.py
data = df

# --- 3. Split Data ---
train_df, test_df = train_test_split(
    data,
    test_size=0.2,
    random_state=0,
    stratify=data[label_col]
)

print(f"Training set size: {len(train_df)}")
print(f"Testing set size: {len(test_df)}")

train_data = TabularDataset(train_df)
test_data = TabularDataset(test_df)

# --- 4. Initialize and Train the Tabular Predictor ---
save_path = 'AutogluonModels/sentimentAnalysis'

if os.path.exists(save_path):
    print(f"Removing existing model directory: {save_path}")
    shutil.rmtree(save_path)

# Initialize the TabularPredictor
predictor = TabularPredictor(
    label=label_col,
    eval_metric='accuracy',
    path=save_path
)

print("--- Starting AutoGluon Tabular training with custom model list... ---")
predictor.fit(
    train_data,
    time_limit=300, # Set a shorter limit for simpler models (5 min)
    presets='medium_quality_faster_train', # Use a preset suitable for tabular data
    
    # We use included_model_types to specify only the simple models we want.
    hyperparameters={
        'LR': {},          # Logistic Regression
        'RF': {},          # Random Forest
        'XGB': {},         # XGBoost
    }
)
print("--- Training complete. ---")

# --- 5. Model Leaderboard and Performance ---
print("\n--- Model Leaderboard (All Trained Models) ---")
leaderboard = predictor.leaderboard(test_data)
print(leaderboard)

# Extract the final ensemble score from the leaderboard
ensemble_score = leaderboard.iloc[0]['score_test']
print(f"\nFinal Ensemble Score ({predictor.eval_metric}): {ensemble_score:.4f}")

# Generate and print the Classification Report
predictions = predictor.predict(test_data)
report = classification_report(
    test_data[label_col],
    predictions
)
print("\n--- Classification Report ---")
print(report)

# --- 6. OPTIMIZE: Keep ALL Models ---
print("\n--- Model Optimization: Keeping ALL Models ---")

# Get the names of all models from the leaderboard
top_model_names = leaderboard['model'].tolist()

print(f"Persisting and cleaning up model directory, keeping only: {top_model_names}")
predictor.delete_models(
    models_to_keep=top_model_names, 
    dry_run=False 
)

print("Model cleanup complete. The saved model folder now contains only the top components.")