import pandas as pd
import random
from itertools import combinations

INPUT_FILE = 'raw_results.csv'
OUTPUT_FILE = 'blind_pool_cross_prompt.csv'

df_raw = pd.read_csv(INPUT_FILE)
print(f"File '{INPUT_FILE}' loaded with {len(df_raw)} samples.")

# Preparing Pairs for a full N-vs-N Prompt Level Comparison
cross_prompt_level_pool_list = []
# Group by everything except prompt_level to compare levels for the same question/model
keys = ['question_id', 'language', 'sample_n', 'domain', 'model']
grouped = df_raw.groupby(keys)

for name, group in grouped:
    # Creates all possible combinations of prompt levels (pairs)
    # ex: [(structured, unstructured), (structured, cot), (unstructured, cot)]
    prompt_level_pairs = list(combinations(group['prompt_level'].unique(), 2))
    
    for prompt_level_A_name, prompt_level_B_name in prompt_level_pairs:
        try:
            # Get response for each prompt level in the pair
            response_A = group[group['prompt_level'] == prompt_level_A_name]['response'].iloc[0]
            response_B = group[group['prompt_level'] == prompt_level_B_name]['response'].iloc[0]
        except IndexError:
            # This might happen if a group is missing a response for a prompt level
            continue

        if pd.isna(response_A) or pd.isna(response_B):
            continue

        # Unpack identifiers from the group name
        question_id, language, sample_n, domain, model_name = name

        # Fixes the seed using unique identifiers from the comparison.
        # This ensures that the A vs. B shuffling is random but reproducible.
        seed_value = hash(str(name)) + hash(prompt_level_A_name) + hash(prompt_level_B_name)
        random.seed(seed_value)
        
        # Shuffle which prompt response will be 'A' and which will be 'B'
        prompts_to_shuffle = [
            (response_A, prompt_level_A_name),
            (response_B, prompt_level_B_name)
        ]
        random.shuffle(prompts_to_shuffle)
        
        cross_prompt_level_pool_list.append({
            'evaluation_id': f"{question_id}_{sample_n}_{domain}_{model_name}_{prompt_level_A_name}_vs_{prompt_level_B_name}",
            'question_id': question_id,
            'sample_n': sample_n,
            'domain': domain,
            'language': language,
            'model': model_name,
            'comparison': f"{prompt_level_A_name}_vs_{prompt_level_B_name}",
            'response_A': prompts_to_shuffle[0][0],
            'prompt_level_of_A': prompts_to_shuffle[0][1],
            'response_B': prompts_to_shuffle[1][0],
            'prompt_level_of_B': prompts_to_shuffle[1][1],
        })

df_cross_prompt_level_pool = pd.DataFrame(cross_prompt_level_pool_list)
df_cross_prompt_level_pool.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

print("Blind pool for cross-prompt-level comparison successfully created!")
print(f"The file '{OUTPUT_FILE}' contains {len(df_cross_prompt_level_pool)} pairs for evaluation.")

