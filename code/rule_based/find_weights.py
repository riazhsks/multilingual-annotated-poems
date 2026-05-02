import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed


class UnifiedPoetryClassifier:
    """
    Classifies poems into poetic forms by comparing handcrafted rules and 
    statistical data (shared vs language-specific) using weighted features.
    """
    def __init__(self, stats_dir, handcrafted_file):
        self.stats_dir = stats_dir
        self.forms_data = self._load_all_forms()
        with open(handcrafted_file, 'r', encoding='utf-8') as f:
            self.handcrafted = json.load(f)

    def _load_all_forms(self):
        forms = {}
        for filename in os.listdir(self.stats_dir):
            if filename.endswith(".json"):
                form_name = filename.replace(".json", "")
                with open(os.path.join(self.stats_dir, filename), 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                    data = raw_data.get(form_name, raw_data)
                    processed = {}
                    for feat, content in data.items():
                        if isinstance(content, dict):
                            processed[feat] = {
                                "shared_values": set(map(str, content.get("shared_values", []))),
                                "eng_only": list(map(str, content.get("eng_only", []))),
                                "cz_only": list(map(str, content.get("cz_only", [])))
                            }
                    forms[form_name] = processed
        return forms

    def classify(self, poem_features, feature_weights, lang):
        """
        Calculates confidence scores for every known form based on weighted feature matches.
        """
        results = {}
        source_multipliers = {"handcrafted": 1.0, "shared": 0.9, "top": 0.6, "rare": 0.3}
        possible_score = sum(feature_weights.values())

        for form_name, stats in self.forms_data.items():
            score = 0

            handcrafted_rule = self.handcrafted[form_name]

            for f, w in feature_weights.items():
                val = str(poem_features.get(f, ""))
                f_stats = stats.get(f, {})
                rule_vals = handcrafted_rule.get(f, [])

                if not isinstance(rule_vals, list): rule_vals = [rule_vals]
                
                if val in map(str, rule_vals):
                    score += (w * source_multipliers["handcrafted"])

                elif val in f_stats.get("shared_values", set()): 
                    score += (w * source_multipliers["shared"])

                else:
                    if lang == "both":
                        eng_list =  f_stats.get("eng_only", [])
                        cz_list =  f_stats.get("cz_only", [])

                        if val in eng_list[:100] or val in cz_list[:100]:
                            # Frequent variant
                            score += (w * source_multipliers["top"])
                        elif val in eng_list or val in cz_list:
                            # Rare variant (The Penalty)
                            score += (w * source_multipliers["rare"])
                    else:
                        target_list = f_stats.get("eng_only", []) if lang == "czech" else f_stats.get("cz_only", [])

                        if val in target_list[:100]:
                            # Frequent variant
                            score += (w * source_multipliers["top"])
                        elif val in target_list:
                            # Rare variant (The Penalty)
                            score += (w * source_multipliers["rare"])

            results[form_name] = round(score / possible_score, 4) if possible_score > 0 else 0
        return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))


def evaluate_config(config, samples, stats_dir, handcrafted_file, eng_to_czech_map, eng_forms, czech_forms):
    """
    Evaluates a specific set of feature weights against train or test samples.
    """
    classifier_instance = UnifiedPoetryClassifier(stats_dir, handcrafted_file)
    
    weights = {
        "line_count": config[0], "stanza_scheme": config[1], 
        "rhyme_scheme": config[2], "rhyme_scheme_per_stanza": config[3],
        "metrical_foot": config[4], "metrical_foot_count": config[5],
        "total_syllables": config[6], "average_syllable_count": config[7]
    }

    correct = 0
    total = 0
    prediction_details = [] 

    for lang in ["czech", "english"]:
        for poem in samples[lang]:
            gt = poem.get("form")
            total += 1
            
            all_confidences = classifier_instance.classify(poem, weights, lang)
            if not all_confidences:
                continue
            
            top_pred = list(all_confidences.keys())[0]
            top_conf = all_confidences[top_pred]

            is_match = False
            if lang == "czech":
                if gt in eng_to_czech_map.get(top_pred, []): is_match = True
            else:
                if gt == top_pred: is_match = True
            
            is_low_conf = False
            relevant_forms = czech_forms if lang == "czech" else eng_forms
            if top_conf < 0.5 and gt not in relevant_forms:
                is_low_conf = True

            was_correct = is_match or is_low_conf
            if was_correct:
                correct += 1

            display_label = "unfixed" if is_low_conf else (top_pred if is_match else "mismatch")
            rounded_conf = {k: round(v, 4) for k, v in all_confidences.items()}
            
            prediction_details.append({
                "poem_id": poem.get("poem_id", "unknown"),
                "lang": lang,
                "gt": gt,
                "prediction": f"{display_label} ({rounded_conf})",
                "is_correct": was_correct
            })

    accuracy = correct / total if total > 0 else 0
    return accuracy, weights, prediction_details


def get_train_test_samples(path, exclusion_file=None, train_ratio=0.8):
    """Loads JSONL data, filters excluded IDs, and splits into train/test sets."""
    excluded_ids = set()
    if exclusion_file and os.path.exists(exclusion_file):
        with open(exclusion_file, 'r', encoding='utf-8') as f:
            excluded_ids = {line.strip() for line in f if line.strip()}
    
    with open(path, 'r', encoding='utf-8') as f:
        filtered_lines = []
        for line in f:
            if not line.strip(): continue
            poem = json.loads(line)
            
            poem_identifier = poem.get("poem_id") 
            
            if poem_identifier not in excluded_ids:
                filtered_lines.append(poem)

        random.shuffle(filtered_lines)
        split_index = int(len(filtered_lines) * train_ratio)
        
        return filtered_lines[:split_index], filtered_lines[split_index:]
    

def run_grid_search(train_data,test_data,sampled_combos):
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(evaluate_config, combo, train_data, STATS_DIR, HANDCRAFTED, 
                                   eng_to_czech_map, eng_forms, czech_forms) 
                   for combo in sampled_combos]
        
        count = 0
        for future in as_completed(futures):
            try:
                acc, weights, _ = future.result()
                count += 1
                if acc >= 0.90:
                    with open("high_accuracy_weights.jsonl", "a") as f:
                        f.write(json.dumps({"accuracy": acc, "weights": weights}) + "\n")
                if count % 100 == 0:
                    print(f"Tested {count}/{combos_size}...")
            except Exception as e:
                print(f"Worker failed with error: {e}")

    results_summary = []
    
    # Evaluate on test data
    if os.path.exists("high_accuracy_weights.jsonl"):
        all_candidates = []
        with open("high_accuracy_weights.jsonl", "r") as f:
            for line in f:
                if line.strip():
                    all_candidates.append(json.loads(line))

        top_8_candidates = sorted(all_candidates, key=lambda x: x['accuracy'], reverse=True)[:8]

        for data in top_8_candidates:
            weights_dict = data["weights"]
            train_acc = data["accuracy"]
            
            weights_list = [
                weights_dict["line_count"], weights_dict["stanza_scheme"],
                weights_dict["rhyme_scheme"], weights_dict["rhyme_scheme_per_stanza"],
                weights_dict["metrical_foot"], weights_dict["metrical_foot_count"],
                weights_dict["total_syllables"], weights_dict["average_syllable_count"]
            ]

            # Run evaluation on the unseen test_data
            test_acc, _, detailed_preds = evaluate_config(
                weights_list, test_data, STATS_DIR, HANDCRAFTED, 
                eng_to_czech_map, eng_forms, czech_forms
            )
            
            results_summary.append({
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "weights": weights_dict,
                "predictions": detailed_preds
            })

        results_summary = sorted(results_summary, key=lambda x: x['test_accuracy'], reverse=True)
        with open("final_test_evaluation_report_detailed.json", "w") as f:
            json.dump(results_summary, f, indent=4)
            

eng_to_czech_map = {
    "sonnet": ["sonet", "sonet_anglický"],
    "limerick": ["limerik"],
    "ghazal": ["gazel"],
    "sestina": ["sestina"],
    "couplet": ["hrdinský_kuplet"]
}

czech_forms = ["sonet", "sonet_anglický","limerik", "gazel", "sestina","hrdinský_kuplet"]
eng_forms = ["sonnet", "limerick", "ghazal", "sestina", "couplet"]

 
if __name__ == "__main__":
    STATS_DIR = 'multilingual_intersection'
    HANDCRAFTED = 'handcrafted_rules.json'
    ENG = "../data_processing/english_data/english_poems_processed.jsonl"
    ENG_SHUFF =  "../shuffled_data/shuffled_samples/english_shuffled.jsonl"
    CZ = "../data_processing/czech_data/czech_poems_processed_ccv.jsonl"
    CZ_SHUFF = "../shuffled_data/shuffled_samples/czech_ccv_shuffled.jsonl"


    eng_train, eng_test = get_train_test_samples(ENG,ENG_SHUFF)
    cz_train, cz_test = get_train_test_samples(CZ, CZ_SHUFF)
    
    train_data = {"english": eng_train, "czech": cz_train}
    test_data = {"english": eng_test, "czech": cz_test}

    # Find best weights configurations on train data with grid-serach
    combos_size = 5000
    val_range = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40]
    sampled_combos = [tuple(random.choice(val_range) for _ in range(8)) for _ in range(combos_size)]
    run_grid_search(train_data,test_data,sampled_combos)

   