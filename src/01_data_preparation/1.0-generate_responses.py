import os
import time
import pandas as pd
import threading
import google.generativeai as genai

from groq import Groq
from tqdm import tqdm
from openai import OpenAI

# --- CONFIGURATION ---
# Ensure your API keys are set as environment variables before running.
# export GEMINI_API_KEY="your_key_here"
# export OPENAI_API_KEY="your_key_here"
# export GROQ_API_KEY="your_key_here"
# export MARITACA_API_KEY="your_key_here"

# Gemini Configuration
gemini_api_key_found = 'GEMINI_API_KEY' in os.environ
if gemini_api_key_found:
    try:
        genai.configure(api_key=os.environ['GEMINI_API_KEY'])
        print("Gemini API Key configured.")
    except ImportError:
        gemini_api_key_found = False
        print("WARNING: Gemini library not installed (`pip install google-generativeai`). Gemini will be skipped.")
else:
    print("WARNING: GEMINI_API_KEY not found. Experiments with Gemini will be skipped.")

# OpenAI Configuration
try:
    client_openai = OpenAI(api_key=os.environ['OPENAI_API_KEY']) if 'OPENAI_API_KEY' in os.environ else None
    if client_openai:
        print("OpenAI API key configured.")
    else:
        print("WARNING: OPENAI_API_KEY not found. Experiments with GPT will be skipped.")
except ImportError:
    client_openai = None
    print("WARNING: OpenAI library not installed (`pip install openai`).")

# Groq Configuration
try:
    client_groq = Groq(api_key=os.environ['GROQ_API_KEY']) if 'GROQ_API_KEY' in os.environ else None
    if client_groq:
        print("Groq API key configured.")
    else:
        print("WARNING: GROQ_API_KEY not found. Experiments with Llama will be skipped.")
except ImportError:
    client_groq = None
    print("WARNING: Groq library not installed (`pip install groq`).")

# Maritaca Configuration
try:
    client_maritaca = OpenAI(
        api_key=os.environ['MARITACA_API_KEY'],
        base_url="https://chat.maritaca.ai/api"
    ) if 'MARITACA_API_KEY' in os.environ else None
    if client_maritaca:
        print("Maritaca API Key configured.")
    else:
        print("WARNING: MARITACA_API_KEY not found. Experiments with Maritaca will be skipped.")
except ImportError:
    client_maritaca = None
    print("WARNING: OpenAI library not installed (`pip install openai`) - required for Maritaca.")

MODELS_TO_TEST = [
    'gemini-1.5-pro-latest',
    'gpt-4o',
    'llama-3.3-70b-versatile',
    'sabia-3.1',
]

GENERATION_PARAMETERS = {"temperature": 0.7, "n_samples": 5}
INPUT_FILE = '../../data/raw/prompts.csv'
MAIN_OUTPUT_FILE = '../../data/raw/raw_results.csv'

file_lock = threading.Lock()
headers_written = set()

def save_result_incrementally(result_data):
    """Saves a single result to the main consolidated output file."""
    with file_lock:
        df_row = pd.DataFrame([result_data])

        write_header = not MAIN_OUTPUT_FILE.exists()
        df_row.to_csv(MAIN_OUTPUT_FILE, mode='a', header=write_header, index=False, encoding='utf-8-sig')

def generate_and_save_responses(prompt_text, model_name, params, context, pbar):
    """Calls the appropriate model API, saves the result, and updates the progress bar."""
    n_samples = params["n_samples"]

    for i in range(n_samples):
        start_time = time.time()
        response_text, total_tokens, error = None, 0, None

        try:
            if "gemini" in model_name:
                if not gemini_api_key_found: raise ConnectionError("Gemini skipped (API key not configured).")
                model = genai.GenerativeModel(model_name)
                generation_config = genai.types.GenerationConfig(temperature=params["temperature"])
                response = model.generate_content(prompt_text, generation_config=generation_config)
                response_text = response.text
                total_tokens = response.usage_metadata.total_token_count

            elif "gpt" in model_name:
                if not client_openai: raise ConnectionError("OpenAI skipped (API key not configured).")
                chat_completion = client_openai.chat.completions.create(
                    messages=[{"role": "user", "content": prompt_text}], model=model_name, temperature=params["temperature"])
                response_text = chat_completion.choices[0].message.content
                total_tokens = chat_completion.usage.total_tokens

            elif "llama" in model_name or "mixtral" in model_name:
                if not client_groq: raise ConnectionError("Groq skipped (API key not configured).")
                chat_completion = client_groq.chat.completions.create(
                    messages=[{"role": "user", "content": prompt_text}], model=model_name, temperature=params["temperature"])
                response_text = chat_completion.choices[0].message.content
                total_tokens = chat_completion.usage.total_tokens

            elif "sabia" in model_name:
                if not client_maritaca: raise ConnectionError("Maritaca skipped (API key not configured).")
                chat_completion = client_maritaca.chat.completions.create(
                    messages=[{"role": "user", "content": prompt_text}],
                    model=model_name,
                    temperature=params["temperature"]
                )
                response_text = chat_completion.choices[0].message.content
                total_tokens = chat_completion.usage.total_tokens
            else:
                error = f"Model '{model_name}' not recognized."

        except Exception as e:
            error = str(e)

        duration = time.time() - start_time

        result = {
            **context,
            'sample_n': i + 1, 'response': response_text,
            'duration_s': round(duration, 2), 'total_tokens': total_tokens, 'error': error
        }

        save_result_incrementally(result)
        pbar.update(1)
        # A small delay can help prevent rate limiting issues
        time.sleep(1.5)

def process_question(question_row, pbar):
    """Processes all experiments for a single question from the input CSV."""
    prompt_levels = ['minimum', 'contextual', 'detailed', 'structured']

    for model in MODELS_TO_TEST:
        pbar.set_description(f"Q_ID: {question_row['id_pergunta']}, Model: {model.split('-')[0]}")
        for language in ['pt', 'en']:
            for level in prompt_levels:
                prompt_column = f'prompt_{level}_{language}'
                if prompt_column in question_row and pd.notna(question_row[prompt_column]):
                    current_prompt = question_row[prompt_column]
                    context_for_prompt = {
                        'question_id': question_row['id_pergunta'], 'domain': question_row['dominio'],
                        'model': model, 'language': language, 'prompt_level': level,
                        'prompt_used': current_prompt
                    }
                    generate_and_save_responses(current_prompt, model, GENERATION_PARAMETERS, context_for_prompt, pbar)

def main():
    """Main function to run the script."""
    print(f"\n--- STARTING RESPONSE GENERATION ---")

    try:
        df_plan = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"ERROR: File '{INPUT_FILE}' not found. Please create it first.")
        return

    print("Calculating total number of API calls...")
    total_api_calls = 0
    prompt_levels = ['minimum', 'contextual', 'detailed', 'structured']
    for _, row in df_plan.iterrows():
        for _ in MODELS_TO_TEST:
            for language in ['pt', 'en']:
                for level in prompt_levels:
                    prompt_column = f'prompt_{level}_{language}'
                    if prompt_column in row and pd.notna(row[prompt_column]):
                        total_api_calls += GENERATION_PARAMETERS['n_samples']

    print(f"Total calls to be made: {total_api_calls}")
    print("-" * 30)

    if total_api_calls == 0:
        print("No prompts found to process. Exiting.")
        return

    with tqdm(total=total_api_calls, desc="Initializing...") as pbar:
        for _, row in df_plan.iterrows():
            process_question(row, pbar)
        pbar.set_description("All tasks completed")

    print("\n--- All questions processed. ---")
    print("All results have been saved incrementally.")
    print(f"Consolidated file: '{MAIN_OUTPUT_FILE}'")
    print("--- END OF SCRIPT ---")

if __name__ == '__main__':
    main()
