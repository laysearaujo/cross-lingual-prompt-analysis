import pandas as pd
import random
import time
import re
from deep_translator import GoogleTranslator

INPUT_FILE = '../../data/processed/sample_master.csv'
OUTPUT_FILE = '../../data/processed/blind_pool_cross_lingual_pt.csv'

def _translate_chunk(text, dest_lang, src_lang):
    """Translates an individual chunk with retry logic"""
    for attempt in range(5):
        try:
            time.sleep(1)
            translator = GoogleTranslator(source=src_lang, target=dest_lang)
            translated = translator.translate(text)
            return translated
        except Exception as e:
            print(f"Chunk error (attempt {attempt + 1}/5): {e}")
            if attempt < 4:
                time.sleep(2)
    return None

def translate_text(text, dest_lang='pt', src_lang='en', max_chars=4500):
    """Translates long texts by splitting them into smaller chunks"""

    if not isinstance(text, str) or pd.isna(text):
        print("Invalid input. Skipping translation.")
        return None

    # Additional sanitization
    text_sanitized = text.replace('`', "'").strip()
    
    # If the text is short, translate directly
    if len(text_sanitized) <= max_chars:
        print(f"    - Translating text ({len(text_sanitized)} chars)...")
        return _translate_chunk(text_sanitized, dest_lang, src_lang)
    
    # If it is long, divide it into paragraphs
    print(f"    - Long text ({len(text_sanitized)} chars). Splitting into chunks...")
    paragraphs = text_sanitized.split('\n\n')
    
    translated_parts = []
    current_chunk = ""
    chunk_count = 0
    
    for para in paragraphs:
        # If adding this paragraph exceeds the limit
        if len(current_chunk) + len(para) + 2 > max_chars:
            if current_chunk:
                chunk_count += 1
                print(f"Translating chunk {chunk_count} ({len(current_chunk)} chars)...")
                translated = _translate_chunk(current_chunk.strip(), dest_lang, src_lang)
                if translated:
                    translated_parts.append(translated)
                else:
                    return None
                current_chunk = ""

            # If a single paragraph is too large, split by sentences
            if len(para) > max_chars:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 > max_chars:
                        if current_chunk:
                            chunk_count += 1
                            print(f"Translating chunk {chunk_count} ({len(current_chunk)} chars)...")
                            translated = _translate_chunk(current_chunk.strip(), dest_lang, src_lang)
                            if translated:
                                translated_parts.append(translated)
                            else:
                                return None
                            current_chunk = ""
                    current_chunk += sentence + " "
            else:
                current_chunk = para + "\n\n"
        else:
            current_chunk += para + "\n\n"

    # Translate the last chunk
    if current_chunk.strip():
        chunk_count += 1
        print(f"Translating final chunk {chunk_count} ({len(current_chunk)} chars)...")
        translated = _translate_chunk(current_chunk.strip(), dest_lang, src_lang)
        if translated:
            translated_parts.append(translated)
        else:
            return None
    
    print(f"Translation complete! {chunk_count} chunks processed.")
    return "\n\n".join(translated_parts)

# --- MAIN LOGIC ---

print(f"Loading master file: '{INPUT_FILE}'...")

df_master = pd.read_csv(INPUT_FILE)
print(f"File loaded with {len(df_master)} samples.")

# Filter only for cross-lingual analysis pairs
df_cross_lingual = df_master[df_master['evaluation_id'].str.endswith('_en_vs_pt', na=False)].copy()

if df_cross_lingual.empty:
    print("No cross-lingual analysis pairs ('_en_vs_pt') found in the master file.")
    exit()

# Prepare the new list of pairs for Portuguese comparison
cross_lingual_pt_pool_list = []

total_pairs = len(df_cross_lingual)
print(f"\nFound {total_pairs} cross-lingual pairs. Processing to create Portuguese analysis set.")

for i, row in df_cross_lingual.iterrows():
    print(f"\n- Processing pair {i+1}/{total_pairs} (ID: {row['evaluation_id']})")

    response_en_original = None
    source_a = str(row.get('source_of_A', '')).strip().casefold()
    source_b = str(row.get('source_of_B', '')).strip().casefold()

    if source_a == 'english_original':
        response_en_original = row['response_A']
    elif source_b == 'english_original':
        response_en_original = row['response_B']
    else:
        print(f" Could not identify 'english_original' in source_of_A ('{row.get('source_of_A')}') or source_of_B ('{row.get('source_of_B')}'). Skipping.")
        continue
    
    response_pt_original = row.get('original_pt_response')

    if pd.isna(response_pt_original) or pd.isna(response_en_original):
        print("One of the original responses (EN or PT) is empty after identification. Skipping.")
        continue

    # Translate the original English response to Portuguese
    print("Translating EN -> PT...")
    response_en_pivoted_to_pt = translate_text(response_en_original)
    
    if not response_en_pivoted_to_pt:
        print("EN->PT translation failed after all attempts. Skipping.")
        continue

    # Set seed to ensure reproducibility
    seed_value = hash(row['evaluation_id'])
    random.seed(seed_value)
    
    # Shuffle which Portuguese response will be 'A' and which will be 'B'
    responses_to_shuffle_pt = [
        (response_pt_original, 'portuguese_original'),
        (response_en_pivoted_to_pt, 'english_pivoted_to_pt')
    ]
    random.shuffle(responses_to_shuffle_pt)
    
    cross_lingual_pt_pool_list.append({
        'evaluation_id': row['evaluation_id'],
        'question_id': row['question_id'],
        'model': row.get('model'),
        'sample_n': row.get('sample_n'),
        'domain': row.get('domain'),
        'prompt_level': row.get('prompt_level'),
        'response_A': responses_to_shuffle_pt[0][0],
        'source_of_A': responses_to_shuffle_pt[0][1],
        'response_B': responses_to_shuffle_pt[1][0],
        'source_of_B': responses_to_shuffle_pt[1][1],
    })

df_cross_lingual_pt_pool = pd.DataFrame(cross_lingual_pt_pool_list)
df_cross_lingual_pt_pool.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

print("\n--- Process Completed ---")
print(f"The file '{OUTPUT_FILE}' contains {len(df_cross_lingual_pt_pool)} pairs for evaluation.")