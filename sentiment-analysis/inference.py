import pandas as pd
from autogluon.tabular import TabularPredictor

# 1. Load the trained model
# AutoGluon saves the best model automatically to this folder
save_path = 'AutogluonModels/sentimentAnalysis' 

print(f"Loading model from {save_path}...")
try:
    predictor = TabularPredictor.load(save_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Did you run model.py to train the model first?")
    exit()

# 2. Define new data to predict
# We use the same column name 'text' as in training
new_data = pd.DataFrame({
    'text': [
        "I absolutely love this new feature! It's amazing.",
        "This is the worst experience I've ever had.",
        "It's okay, nothing special.",
        "I'm not sure how I feel about this."
    ]
})

print("\nInput Data:")
print(new_data)

# 3. Make predictions
# The model will output -1 (Negative), 0 (Neutral), or 1 (Positive)
predictions = predictor.predict(new_data)

print("\nPredictions (-1=Neg, 0=Neu, 1=Pos):")
print(predictions)

# 4. (Optional) Get probabilities for each class
probs = predictor.predict_proba(new_data)
print("\nPrediction Probabilities:")
print(probs)
