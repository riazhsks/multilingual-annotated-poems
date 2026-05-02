import json
import os
from collections import Counter, defaultdict

def load_json(file_path):
    """Helper to load JSON data."""
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    """Helper to save JSON data with formatting."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def process_poetic_data(input_file, output_dir):
    """Aggregates poetic features by form and saves individual JSON summaries."""
    os.makedirs(output_dir, exist_ok=True)
    
    stats = defaultdict(lambda: defaultdict(Counter))
    features_to_track = [
        "stanza_scheme", "rhyme_scheme", "rhyme_scheme_per_stanza",
        "metrical_foot", "metrical_foot_count", "line_count",
        "total_syllables", "average_syllable_count"
    ]

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            
            data = json.loads(line)
            form = data.get("form")
            if not form: continue

            for feat in features_to_track:
                val = data.get(feat)
                if val is not None:
                    stats[form][feat][val] += 1

    for form_name, features in stats.items():
        output = {form_name: {feat: dict(cnt) for feat, cnt in features.items()}}
        
        safe_name = form_name.replace(" ", "_")
        save_json(output, os.path.join(output_dir, f"{safe_name}.json"))
    

def generate_per_form_intersections(lang_a_dir, lang_b_dir, mapping, output_dir):
    """Computes shared and unique feature values between two languages based on a mapping."""
    os.makedirs(output_dir, exist_ok=True)

    for eng_form, czech_forms in mapping.items():
        eng_raw = load_json(os.path.join(lang_a_dir, f"{eng_form}.json"))
        if not eng_raw:
            print(f"Warning: {eng_form} data not found in {lang_a_dir}")
            continue
        eng_data = eng_raw.get(eng_form, {})

        combined_czech = defaultdict(set)
        for cz_form in czech_forms:
            cz_raw = load_json(os.path.join(lang_b_dir, f"{cz_form}.json"))
            if not cz_raw: continue
            
            cz_data = cz_raw.get(cz_form, {})
            for feat, values_dict in cz_data.items():
                combined_czech[feat].update(values_dict.keys())

        form_intersection = {}
        for feat, eng_values_dict in eng_data.items():
            eng_set = set(eng_values_dict.keys())
            cz_set = combined_czech.get(feat, set())
            
            shared = eng_set.intersection(cz_set)
            eng_only = eng_set - cz_set
            cz_only = cz_set - eng_set

            form_intersection[feat] = {
                "shared_values": sorted(list(shared)),
                "count_shared": len(shared),
                "eng_only": sorted(list(eng_only)),
                "eng_only_length": len(eng_only),
                "cz_only": sorted(list(cz_only)),
                "cz_only_length": len(cz_only)
            }
        
        if form_intersection:
            output_path = os.path.join(output_dir, f"{eng_form.replace(' ', '_')}.json")
            save_json(form_intersection, output_path)

if __name__ == "__main__":

    process_poetic_data("../data_processing/czech_data/czech_poems_processed_ccv.jsonl", 
        'form_statistics_czech')
    process_poetic_data("../data_processing/english_data/english_poems_processed.jsonl", 
        'form_statistics_english' )

    same_forms_compared = {
        "sonnet": ["sonet", "sonet_anglický"],
        "limerick": ["limerik"],
        "ghazal": ["gazel"],
        "sestina": ["sestina"],
        "couplet": ["hrdinský_kuplet"]
    }

    generate_per_form_intersections(
        'form_statistics_english', 
        'form_statistics_czech', 
        same_forms_compared, 
        'multilingual_intersection'
    )