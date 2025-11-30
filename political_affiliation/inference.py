import pandas as pd
from autogluon.multimodal import MultiModalPredictor
# from autogluon.tabular import TabularPredictor # Uncomment if you want to use the tabular model

# 1. Load the trained model
# We default to the Transformer model as it's usually better for text
save_path = 'AutogluonModels/politicalAffiliation_transformer' 

print(f"Loading model from {save_path}...")
try:
    predictor = MultiModalPredictor.load(save_path)
    # predictor = TabularPredictor.load('AutogluonModels/politicalAffiliation_tabular') # For tabular
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Did you run model.py to train the model first?")
    exit()

# 2. Define new data to predict
new_data = pd.DataFrame({
    'Tweet': [
        "We need to lower taxes and support small businesses.",
        "Climate change is a crisis we must address immediately.",
        "The constitution protects our rights.",
        "Universal healthcare is a human right."
    ]
})

print("\nInput Data:")
print(new_data)

# 3. Make predictions
predictions = predictor.predict(new_data)

print("\nPredictions:")
print(predictions)

# 4. Probabilities
probs = predictor.predict_proba(new_data)
print("\nProbabilities:")
print(probs)
