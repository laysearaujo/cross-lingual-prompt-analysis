import pandas as pd
import random
from itertools import combinations

INPUT_FILE = '../../data/raw/raw_results.csv'
OUTPUT_FILE = '../../data/processed/blind_pool_cross_model.csv'\

df_raw = pd.read_csv(INPUT_FILE)
print(f"File '{INPUT_FILE}' loaded with {len(df_raw)} samples.")

# Preparing Pairs for Model Comparison
cross_model_pool_list = []
keys = ['question_id', 'language', 'sample_n', 'domain', 'prompt_level']

# Filters only the prompt level we want to compare and groups (1:n)
grouped = df_raw.groupby(keys)

for name, group in grouped:
    # Creates all possible combinations of models (pairs)
    # ex: [(gemini, gpt), (gemini, llama), (gpt, llama)]
    model_pairs = list(combinations(group['model'].unique(), 2))
    
    for model_A_name, model_B_name in model_pairs:
        try:
            response_A = group[group['model'] == model_A_name]['response'].iloc[0]
            response_B = group[group['model'] == model_B_name]['response'].iloc[0]
        except IndexError:
            continue

        if pd.isna(response_A) or pd.isna(response_B):
            continue

        # Fixes the seed using unique identifiers from the comparison (question ID, models, etc.).
        # This ensures that the A vs. B shuffling is random but reproducible,
        # mitigating the LLM-Judge order bias.
        seed_value = hash(str(name)) + hash(model_A_name) + hash(model_B_name)
        random.seed(seed_value)
        
        # Shuffle which model will be 'A' and which will be 'B'
        models_to_shuffle = [
            (response_A, model_A_name),
            (response_B, model_B_name)
        ]
        random.shuffle(models_to_shuffle)
        
        cross_model_pool_list.append({
            'evaluation_id': f"{name[0]}_{name[2]}_{name[3]}_{name[4]}_{model_A_name}_vs_{model_B_name}" + (f"_{name[1]}" if name[1] else ""),
            'question_id': name[0],
            'prompt_level': name[4],
            'sample_n': name[2],
            'domain': name[3],
            'language': name[1],
            'comparison': f"{model_A_name}_vs_{model_B_name}",
            'response_A': models_to_shuffle[0][0],
            'model_of_A': models_to_shuffle[0][1],
            'response_B': models_to_shuffle[1][0],
            'model_of_B': models_to_shuffle[1][1],
        })

df_cross_model_pool = pd.DataFrame(cross_model_pool_list)
df_cross_model_pool.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

print("Blind pool for cross-model comparison successfully created!")
print(f"The file '{OUTPUT_FILE}' contains {len(df_cross_model_pool)} pairs for evaluation.")