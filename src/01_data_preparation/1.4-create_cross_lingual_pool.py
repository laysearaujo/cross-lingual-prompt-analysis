import pandas as pd
import random
from googletrans import Translator

INPUT_FILE = '../../data/raw/raw_results.csv'
OUTPUT_FILE = '../../data/processed/blind_pool_cross_lingual.csv'

# Pivot Translation is a technique used to compare texts from different languages ​​fairly
# It avoids the bias of asking an LLM to directly compare texts in different languages.
def translate_text_to_english(text):
    """Translates a text into English using the Google Translate API."""
    if not text or pd.isna(text):
        return None
    try:
        translator = Translator()
        translated = translator.translate(text, src='pt', dest='en')
        return translated.text
    except Exception as e:
        print(f"Translation error: {e}")
        return None

df_raw = pd.read_csv(INPUT_FILE)
print(f"File '{INPUT_FILE}' loaded with {len(df_raw)} samples.")

# Preparing Pairs for Model Comparison
cross_lingual_pool_list = []
keys = ['question_id', 'model', 'sample_n', 'domain', 'prompt_level']

# Filters only the prompt level we want to compare and groups (n:n)
grouped = df_raw.groupby(keys)

for name, group in grouped:
    try:
        # Get the original answer in English and the original in Portuguese
        response_en_original = group[group['language'] == 'en']['response'].iloc[0]
        response_pt_original = group[group['language'] == 'pt']['response'].iloc[0]
    except IndexError:
        continue
        
    if pd.isna(response_en_original) or pd.isna(response_pt_original):
        continue

    # Pivot Translation Stage
    print(f"Translating response to {name[0]}...")
    response_pt_pivoted_to_en = translate_text_to_english(response_pt_original)
    
    if not response_pt_pivoted_to_en:
        print(f"Translation failed for {name[0]}. Skipping.")
        continue

    # Fixes the seed using unique identifiers from the comparison.
    seed_value = hash(str(name))
    random.seed(seed_value)
    
    # Shuffle which model will be 'A' and which will be 'B'
    responses_to_shuffle = [
        (response_en_original, 'english_original'),
        (response_pt_pivoted_to_en, 'portuguese_pivoted')
    ]
    random.shuffle(responses_to_shuffle)
    
    cross_lingual_pool_list.append({
        'evaluation_id': f"{name[0]}_{name[1]}_{name[2]}_{name[3]}_{name[4]}_en_vs_pt",
        'question_id': name[0],
        'model': name[1],
        'sample_n': name[2],
        'domain': name[3],
        'prompt_level': name[4],
        'response_A': responses_to_shuffle[0][0],
        'source_of_A': responses_to_shuffle[0][1],
        'response_B': responses_to_shuffle[1][0],
        'source_of_B': responses_to_shuffle[1][1],
        'original_pt_response': response_pt_original,
    })

df_cross_lingual_pool = pd.DataFrame(cross_lingual_pool_list)
df_cross_lingual_pool.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

print("Blind pool for cross-lingual comparison created successfully!")
print(f"The file '{OUTPUT_FILE}' contains {len(df_cross_lingual_pool)} pairs for evaluation.")