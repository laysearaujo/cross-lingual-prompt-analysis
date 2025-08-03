import pandas as pd
import os
import time
from tqdm import tqdm
import threading

# --- CONFIGURATION ---
# Ensure your API keys are set as environment variables before running.
# export GEMINI_API_KEY="your_key_here"
# export OPENAI_API_KEY="your_key_here"
# export GROQ_API_KEY="your_key_here"

try:
    import google.generativeai as genai
    if 'GEMINI_API_KEY' in os.environ:
        genai.configure(api_key=os.environ['GEMINI_API_KEY'])
        print("Gemini API Key configured.")
    else:
        print("WARNING: GEMINI_API_KEY not found. Experiments with Gemini will be skipped.")
except ImportError:
    print("WARNING: Gemini library not installed (`pip install google-generativeai`).")

try:
    from openai import OpenAI
    if 'OPENAI_API_KEY' in os.environ:
        client_openai = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        print("OpenAI API key configured.")
    else:
        client_openai = None
        print("WARNING: OPENAI_API_KEY not found. Experiments with GPT will be skipped.")
except ImportError:
    client_openai = None
    print("WARNING: OpenAI library not installed (`pip install openai`).")

try:
    from groq import Groq
    if 'GROQ_API_KEY' in os.environ:
        client_groq = Groq(api_key=os.environ['GROQ_API_KEY'])
        print("Groq API key configured.")
    else:
        client_groq = None
        print("WARNING: GROQ_API_KEY not found. Experiments with Llama will be skipped.")
except ImportError:
    client_groq = None
    print("WARNING: Groq library not installed (`pip install groq`).")


MODELS_TO_TEST = [
    'gemini-1.5-pro-latest',
    'gpt-4o',
    'llama3-70b-8192'
]
GENERATION_PARAMETERS = {"temperature": 0.7, "n_samples": 5}
INPUT_FILE = 'prompts.csv'
MAIN_OUTPUT_FILE = 'results_consolidated.csv'
DOMAIN_RESULTS_DIR = 'domain_results'

file_lock = threading.Lock()
headers_written = set()

def save_result_incrementally(result_data):
    with file_lock:
        df_row = pd.DataFrame([result_data])
        write_header_main = not os.path.exists(MAIN_OUTPUT_FILE)
        df_row.to_csv(MAIN_OUTPUT_FILE, mode='a', header=write_header_main, index=False, encoding='utf-8-sig')
        
        domain = result_data.get('domain', 'unknown_domain')
        domain_filename = f"results_{domain.replace(' ', '_').lower()}.csv"
        domain_filepath = os.path.join(DOMAIN_RESULTS_DIR, domain_filename)

        if domain_filepath not in headers_written:
             write_header_domain = not os.path.exists(domain_filepath)
        else:
            write_header_domain = False

        df_row.to_csv(domain_filepath, mode='a', header=write_header_domain, index=False, encoding='utf-8-sig')
        
        if write_header_domain:
            headers_written.add(domain_filepath)

def generate_and_save_responses(prompt_text, model_name, params, context, pbar):
    """
    Calls the API, saves the result, and updates the shared progress bar.
    """
    n_samples = params["n_samples"]
    
    for i in range(n_samples):
        start_time = time.time()
        response_text, total_tokens, error = None, 0, None

        try:
            if "gemini" in model_name:
                if 'GEMINI_API_KEY' not in os.environ: raise ConnectionError("Gemini skipped")
                model = genai.GenerativeModel(model_name)
                generation_config = genai.types.GenerationConfig(temperature=params["temperature"])
                response = model.generate_content(prompt_text, generation_config=generation_config)
                response_text = response.text
                total_tokens = response.usage_metadata.total_token_count

            elif "gpt" in model_name:
                if not client_openai: raise ConnectionError("OpenAI skipped")
                chat_completion = client_openai.chat.completions.create(
                    messages=[{"role": "user", "content": prompt_text}], model=model_name, temperature=params["temperature"])
                response_text = chat_completion.choices[0].message.content
                total_tokens = chat_completion.usage.total_tokens

            elif "llama" in model_name:
                if not client_groq: raise ConnectionError("Groq skipped")
                chat_completion = client_groq.chat.completions.create(
                    messages=[{"role": "user", "content": prompt_text}], model=model_name, temperature=params["temperature"])
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
            'duration_s': duration, 'total_tokens': total_tokens, 'error': error
        }
        
        save_result_incrementally(result)
        pbar.update(1)
        time.sleep(1.5)

def process_question(question_row, pbar):
    """
    Processes all experiments for a single question, passing the progress bar down.
    """
    prompt_levels = ['base', 'contextual', 'detalhado', 'estruturado']
    
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

if __name__ == '__main__':
    print(f"\n--- STARTING RESPONSE GENERATION ---")
    
    if not os.path.exists(DOMAIN_RESULTS_DIR):
        os.makedirs(DOMAIN_RESULTS_DIR)

    try:
        df_plan = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"ERROR: File '{INPUT_FILE}' not found. Please create it first.")
        exit()

    print("Calculating total number of API calls...")
    total_api_calls = 0
    prompt_levels = ['base', 'contextual', 'detalhado', 'estruturado']
    for index, row in df_plan.iterrows():
        for model in MODELS_TO_TEST:
            for language in ['pt', 'en']:
                for level in prompt_levels:
                    prompt_column = f'prompt_{level}_{language}'
                    if prompt_column in row and pd.notna(row[prompt_column]):
                        total_api_calls += GENERATION_PARAMETERS['n_samples']
    
    print(f"Total calls to be made: {total_api_calls}")
    print("-" * 30)

    with tqdm(total=total_api_calls, desc="Initializing...") as pbar:
        for index, row in df_plan.iterrows():
            process_question(row, pbar)
        
        pbar.set_description("All tasks completed")

    print("\n--- All questions processed. ---")
    print(f"All results have been saved incrementally.")
    print(f"Main consolidated file: '{MAIN_OUTPUT_FILE}'")
    print(f"Domain-specific files are in: '{DOMAIN_RESULTS_DIR}/'")
    print("--- END OF SCRIPT ---")