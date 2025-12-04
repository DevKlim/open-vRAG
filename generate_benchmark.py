import pandas as pd
import os
import time
from autogluon.tabular import TabularPredictor
import google.generativeai as genai

# --- Configuration ---
POLITICAL_DATA_PATH = 'political_affiliation/test.tsv'
CLICKBAIT_DATA_PATH = 'clickbaitDataAndTraining/clickbait_data.csv'
SENTIMENT_DATA_PATH = 'sentiment-analysis/training.1600000.processed.noemoticon.csv'

POLITICAL_MODEL_PATH = 'AutogluonModels/politicalAffiliation_tabular' 
CLICKBAIT_MODEL_PATH = 'clickbaitDataAndTraining/ag_models_clickbait_tabular'
SENTIMENT_MODEL_PATH = 'sentiment-analysis/AutogluonModels/sentiment-analysis_tabular'

OUTPUT_MD = 'benchmark_results.md'
OUTPUT_CSV = 'benchmark_results.csv'

SAMPLE_SIZE = 20  # Increased as per user request (implied "20 runs")

# API Key provided by user
API_KEY = "AIzaSyBJfMhWQFUI1Dg3nu3i1WFEWUy-v10nwyc"

# --- Helper Functions ---

def load_political_data(path, n=10):
    print(f"Loading political data from {path}...")
    columns = [
        "id", "label", "statement", "subjects", "speaker", "speaker_job",
        "state", "party", "barely_true_count", "false_count",
        "half_true_count", "mostly_true_count", "pants_on_fire_count", "context"
    ]
    try:
        df = pd.read_csv(path, sep='\t', header=None, names=columns)
        df = df[df['party'].isin(['democrat', 'republican'])]
        return df.sample(n=min(n, len(df)), random_state=42)
    except Exception as e:
        print(f"Error loading political data: {e}")
        return pd.DataFrame()

def load_clickbait_data(path, n=10):
    print(f"Loading clickbait data from {path}...")
    try:
        df = pd.read_csv(path)
        return df.sample(n=min(n, len(df)), random_state=42)
    except Exception as e:
        print(f"Error loading clickbait data: {e}")
        return pd.DataFrame()

def load_sentiment_data(path, n=10):
    print(f"Loading sentiment data from {path}...")
    # Sentiment140 format: target, id, date, flag, user, text
    columns = ["target", "ids", "date", "flag", "user", "text"]
    try:
        df = pd.read_csv(path, encoding='latin-1', header=None, names=columns)
        # Target: 0 = negative, 4 = positive
        return df.sample(n=min(n, len(df)), random_state=42)
    except Exception as e:
        print(f"Error loading sentiment data: {e}")
        return pd.DataFrame()

def get_gemini_response(model, prompt):
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

# --- Main Execution ---

def main():
    # 1. Setup GenAI
    genai.configure(api_key=API_KEY)
    gemini_model = genai.GenerativeModel('gemini-flash-latest')

    # 2. Load Data
    df_pol = load_political_data(POLITICAL_DATA_PATH, SAMPLE_SIZE)
    df_cb = load_clickbait_data(CLICKBAIT_DATA_PATH, SAMPLE_SIZE)
    df_sent = load_sentiment_data(SENTIMENT_DATA_PATH, SAMPLE_SIZE)

    results = []

    # 3. Political Affiliation Benchmark
    print("\n--- Running Political Affiliation Benchmark ---")
    if not df_pol.empty:
        try:
            pred_pol = TabularPredictor.load(POLITICAL_MODEL_PATH)
            print("Loaded Political AutoGluon model.")
        except Exception as e:
            print(f"Failed to load Political AutoGluon model: {e}")
            pred_pol = None

        for _, row in df_pol.iterrows():
            statement = row['statement']
            true_label = row['party']
            
            # PredAI
            pred_ai_out = "N/A"
            if pred_pol:
                try:
                    row_data = row.to_frame().T
                    row_data = row_data.rename(columns={'statement': 'Tweet'})
                    pred_ai_out = pred_pol.predict(row_data).iloc[0]
                except Exception as e:
                    pred_ai_out = f"Error: {e}"

            # GenAI - Direct
            prompt_direct = f"Classify the following statement as 'democrat' or 'republican'. Return ONLY the label.\n\nStatement: {statement}"
            gen_ai_direct = get_gemini_response(gemini_model, prompt_direct)

            # GenAI - CoT
            prompt_cot = f"Classify the following statement as 'democrat' or 'republican'. First, explain your reasoning step-by-step. Then, conclude with 'Label: <label>'.\n\nStatement: {statement}"
            gen_ai_cot = get_gemini_response(gemini_model, prompt_cot)

            results.append({
                "Task": "Political Affiliation",
                "Input": statement[:100] + "..." if len(statement) > 100 else statement,
                "True Label": true_label,
                "PredAI (AutoGluon)": pred_ai_out,
                "GenAI (Direct)": gen_ai_direct,
                "GenAI (CoT)": gen_ai_cot
            })

    # 4. Clickbait Benchmark
    print("\n--- Running Clickbait Benchmark ---")
    if not df_cb.empty:
        try:
            pred_cb = TabularPredictor.load(CLICKBAIT_MODEL_PATH)
            print("Loaded Clickbait AutoGluon model.")
        except Exception as e:
            print(f"Failed to load Clickbait AutoGluon model: {e}")
            pred_cb = None

        for _, row in df_cb.iterrows():
            text_col = 'headline' if 'headline' in df_cb.columns else df_cb.columns[0]
            label_col = 'clickbait' if 'clickbait' in df_cb.columns else df_cb.columns[-1]
            
            text = str(row[text_col])
            true_label = row[label_col]

            # PredAI
            pred_ai_out = "N/A"
            if pred_cb:
                try:
                    pred_ai_out = pred_cb.predict(pd.DataFrame([row])).iloc[0]
                except Exception as e:
                    pred_ai_out = f"Error: {e}"

            # GenAI - Direct
            prompt_direct = f"Is the following headline clickbait? Answer '1' for yes or '0' for no. Return ONLY the number.\n\nHeadline: {text}"
            gen_ai_direct = get_gemini_response(gemini_model, prompt_direct)

            # GenAI - CoT
            prompt_cot = f"Is the following headline clickbait? Analyze the headline's style and content. Then, conclude with 'Label: 1' (yes) or 'Label: 0' (no).\n\nHeadline: {text}"
            gen_ai_cot = get_gemini_response(gemini_model, prompt_cot)

            results.append({
                "Task": "Clickbait Detection",
                "Input": text[:100] + "..." if len(text) > 100 else text,
                "True Label": true_label,
                "PredAI (AutoGluon)": pred_ai_out,
                "GenAI (Direct)": gen_ai_direct,
                "GenAI (CoT)": gen_ai_cot
            })

    # 5. Sentiment Benchmark
    print("\n--- Running Sentiment Analysis Benchmark ---")
    if not df_sent.empty:
        try:
            pred_sent = TabularPredictor.load(SENTIMENT_MODEL_PATH)
            print("Loaded Sentiment AutoGluon model.")
        except Exception as e:
            print(f"Failed to load Sentiment AutoGluon model: {e}")
            pred_sent = None

        for _, row in df_sent.iterrows():
            text = row['text']
            true_label = row['target'] # 0 or 4
            
            # PredAI
            pred_ai_out = "N/A"
            if pred_sent:
                try:
                    # Model likely expects 'text' column or similar.
                    # Based on standard AutoGluon usage, passing the row as DF is safest.
                    # We might need to rename 'text' if the model was trained on a different column name.
                    # Assuming 'text' is correct for now.
                    pred_ai_out = pred_sent.predict(pd.DataFrame([row])).iloc[0]
                except Exception as e:
                    pred_ai_out = f"Error: {e}"

            # GenAI - Direct
            prompt_direct = f"Classify the sentiment of the following tweet as '0' (negative) or '4' (positive). Return ONLY the number.\n\nTweet: {text}"
            gen_ai_direct = get_gemini_response(gemini_model, prompt_direct)

            # GenAI - CoT
            prompt_cot = f"Classify the sentiment of the following tweet as '0' (negative) or '4' (positive). Explain your reasoning. Then, conclude with 'Label: <number>'.\n\nTweet: {text}"
            gen_ai_cot = get_gemini_response(gemini_model, prompt_cot)

            results.append({
                "Task": "Sentiment Analysis",
                "Input": text[:100] + "..." if len(text) > 100 else text,
                "True Label": true_label,
                "PredAI (AutoGluon)": pred_ai_out,
                "GenAI (Direct)": gen_ai_direct,
                "GenAI (CoT)": gen_ai_cot
            })

    # 6. Save Results
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_CSV, index=False)
    
    # Create Markdown Table
    md_content = "# Model Benchmark Results\n\n"
    md_content += df_results.to_markdown(index=False)
    
    with open(OUTPUT_MD, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"\nBenchmark completed. Results saved to {OUTPUT_MD} and {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
