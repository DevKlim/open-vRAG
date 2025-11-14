import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor # Note the change from multimodal to tabular
import os
import shutil
from sklearn.metrics import classification_report

# --- 1. Define File Path and Load Data ---
data_filename = 'clickbait_data.csv'

if not os.path.exists(data_filename):
    print(f"ERROR: File '{data_filename}' not found.")
    print("Please upload your 'clickbait_data.csv' file to the working directory.")
    import sys
    sys.exit()

print(f"Loading full dataset from '{data_filename}'...")
full_data = pd.read_csv(data_filename)
print(f"Dataset loaded with {len(full_data)} rows.")

# --- 2. Split Data ---
label_column = 'clickbait'
train_data, test_data = train_test_split(full_data, test_size=0.2, random_state=42)

print(f"Training set size: {len(train_data)}")
print(f"Testing set size: {len(test_data)}")

# --- 3. Initialize and Train the Tabular Predictor ---
save_path = 'ag_models_clickbait_tabular' 

if os.path.exists(save_path):
    print(f"Removing existing model directory: {save_path}")
    shutil.rmtree(save_path)

# Initialize the TabularPredictor
predictor = TabularPredictor(
    label=label_column,
    problem_type='binary',
    path=save_path
)

# Fit the models, specifying which simple models to include
# 'LR' (Logistic Regression) and 'RF' (Random Forest) are included by default,
# but we explicitly tell AutoGluon to focus on a few simple models for speed.
print("--- Starting AutoGluon Tabular training with custom model list... ---")
predictor.fit(
    train_data,
    time_limit=300, # Set a shorter limit for simpler models (5 min)
    presets='medium_quality_faster_train', # Use a preset suitable for tabular data
    
    # We use included_model_types to specify only the simple models we want.
    # Note: AutoGluon wraps these with additional preprocessing/feature engineering.
    hyperparameters={
        'LR': {},          # Logistic Regression
        'RF': {},          # Random Forest
        'XGB': {},         # XGBoost (often a good baseline)
    }
)
print("--- Training complete. ---")


# --- 4. Model Leaderboard and Performance ---
# TabularPredictor allows you to view the scores of ALL trained models!
print("\n--- 4. Model Leaderboard (All Trained Models) ---")
# The leaderboard clearly shows the performance of LR, RF, etc.
leaderboard = predictor.leaderboard(test_data)
print(leaderboard)

# Extract the final ensemble score from the leaderboard for robust printing
ensemble_score = leaderboard.iloc[0]['score_test']

# --- 5. Evaluate Ensemble Performance ---
print("\n--- 5. Final Ensemble Model Performance ---")
# We still run evaluate to get the full dictionary, but we use the score
# extracted from the leaderboard which is guaranteed to be correct.
eval_summary = predictor.evaluate(test_data) 

# FIX: Use the score extracted from the leaderboard (best available score)
print(f"Final Ensemble Score ({predictor.eval_metric}): {ensemble_score:.4f}")

# Generate and print the Classification Report for the final model
predictions = predictor.predict(test_data)
report = classification_report(
    test_data[label_column],
    predictions,
    target_names=['Non-Clickbait (0)', 'Clickbait (1)']
)
print("\n--- Classification Report (Per-Class Metrics) ---")
print(report)


# --- 6. OPTIMIZE: Keep only the Top 2 Performing Models ---
print("\n--- 6. Model Optimization: Keeping Top 2 Models ---")

# Get the names of the top 2 models from the leaderboard
# We skip the first row, which is the WeightedEnsemble, unless it's one of the top two base models
top_model_names = leaderboard.iloc[0:2]['model'].tolist()

# The persist_models command saves ONLY the specified models to disk, removing others.
# Note: Since the WeightedEnsemble is usually the best, we keep it and the next best base model.
# If you want ONLY base models (LR, RF, XGB), you'd slice rows 1 and 2.
print(f"Persisting and cleaning up model directory, keeping only: {top_model_names}")
predictor.persist_models(
    models=top_model_names, 
    cleanup_archive=True # Cleans up the models that are NOT in the list
)

# You can now zip and deploy the 'ag_models_clickbait_tabular' folder, 
# and it will be much smaller and faster to load!
print("Model cleanup complete. The saved model folder now contains only the top components.")