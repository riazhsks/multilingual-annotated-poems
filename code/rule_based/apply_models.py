import json
import os
import statistics
from collections import Counter
import pandas as pd 
from find_weights import UnifiedPoetryClassifier

def calculate_confidence_stats(lang, weight_idx, confidences, total_lines):
    """Computes a dictionary of statistics for a set of confidences."""
    top_confidences = [c for c in confidences if c >= 0.5]
    
    # Safely handle empty lists to avoid statistics errors
    safe_mean = lambda x: statistics.mean(x) if x else 0
    
    return {
        "Language": lang,
        "Total Poems": total_lines,
        f"Weight set {weight_idx} Top Mean Confidence": safe_mean(top_confidences),
        f"Weight set {weight_idx} Mean Confidence": safe_mean(confidences),
        "Median Confidence": statistics.median(confidences) if confidences else 0,
        "Max Confidence": max(confidences) if confidences else 0,
        "Min Confidence": min(confidences) if confidences else 0,
        "Standard Deviation": statistics.stdev(confidences) if len(confidences) > 1 else 0
    }

def predict(corpora, is_sample, optimal_weights, classifier):
    # Determine base directory and stats file path
    base_dir = "predictions/samples" if is_sample else "predictions"
    os.makedirs(base_dir, exist_ok=True)
    overall_stats_path = os.path.join(base_dir, "identification_statistics.txt")

    all_stats_records = []

    with open(overall_stats_path, "w", encoding="utf-8") as stats_file:
        for lang, corpus_path in corpora.items():
            lang_dir = os.path.join(base_dir, lang)
            os.makedirs(lang_dir, exist_ok=True)

            for i, weights in enumerate(optimal_weights, 1):
                prediction_file = os.path.join(lang_dir, f"set{i}_classification.txt")
                
                confidences = []
                classifications = Counter()
                total_lines = 0

                with open(prediction_file, "w", encoding="utf-8") as out, \
                     open(corpus_path, "r", encoding="utf-8") as f:
                    
                    for line in f:
                        if not line.strip(): continue
                        poem_data = json.loads(line)
                        total_lines += 1
                        preds = classifier.classify(poem_data, weights, "both")

                        if preds:
                            top_form = next(iter(preds))
                            top_conf = preds[top_form]
                            confidences.append(top_conf)

                            label = top_form if top_conf >= 0.5 else "other"
                            classifications[label] += 1
                            out.write(f"{label} {poem_data.get('form')} ({preds})\n")

                # Generate and record stats
                current_stats = calculate_confidence_stats(lang, i, confidences, total_lines)
                all_stats_records.append(current_stats)

                # Write to text summary
                stats_file.write(f"Statistics for {lang} | Weight set {i}\n")
                stats_file.write("--- Overall Confidence Statistics ---\n")
                for k, v in current_stats.items():
                    stats_file.write(f"{k}: {v}\n")

                stats_file.write("\n--- Predicted Form Distribution (Conf >= 0.5) ---\n")
                for form, count in classifications.most_common():
                    stats_file.write(f"{form}: {count}\n")
                stats_file.write("\n\n")

    # Final CSV Report Generation
    generate_csv_report(all_stats_records, base_dir, len(optimal_weights))

def generate_csv_report(records, base_dir, num_weights):
    df_stats = pd.DataFrame(records)
    
    # Identify standard and top weight columns
    w_cols = [f'Weight set {i} Mean Confidence' for i in range(1, num_weights + 1)]
    top_w_cols = [f'Weight set {i} Top Mean Confidence' for i in range(1, num_weights + 1)]
    
    # Pivot/Group by Language
    pivoted = df_stats.groupby('Language', sort=False)[w_cols + top_w_cols].max().reset_index()
    
    # Rename columns to W1, W2... and TopW1, TopW2...
    w_rename = {old: f'W{i+1}' for i, old in enumerate(w_cols)}
    top_w_rename = {old: f'TopW{i+1}' for i, old in enumerate(top_w_cols)}
    pivoted = pivoted.rename(columns={**w_rename, **top_w_rename})

    new_w_names = list(w_rename.values())
    new_top_w_names = list(top_w_rename.values())

    # Calculate Averages
    pivoted['Avg_W'] = pivoted[new_w_names].mean(axis=1)
    pivoted['Avg_TopW'] = pivoted[new_top_w_names].mean(axis=1)

    # Add Summary Row
    avg_series = pivoted.mean(numeric_only=True)
    avg_series['Language'] = 'AVERAGE'
    pivoted = pd.concat([pivoted, avg_series.to_frame().T], ignore_index=True)

    # Round and Save
    pivoted = pivoted.round(2)
    pivoted.to_csv(os.path.join(base_dir, "identification_statistics.csv"), index=False)

if __name__ == "__main__":
    poetry_classifier = UnifiedPoetryClassifier('multilingual_intersection', 'handcrafted_rules.json')
    
    # Configuration
    path_map = {
        "corpora": {
            "CCV": "../data_processing/czech_data/czech_poems_processed_ccv.jsonl",
            "C3P": "../data_processing/czech_data/czech_poems_processed_c3p.jsonl",
            "German": "../data_processing/german_data/german_poems_processed.jsonl",
            "Spanish": "../data_processing/spanish_data/spanish_poems_processed.jsonl",
            "Hungarian": "../data_processing/hungarian_data/hungarian_poems_processed.jsonl"
        },
        "samples": {
            "CCV": "../shuffled_data/shuffled_samples/czech_ccv_shuffled.jsonl",
            "C3P": "../shuffled_data/shuffled_samples/czech_c3p_shuffled.jsonl",
            "English": "../shuffled_data/shuffled_samples/english_shuffled.jsonl",
            "German": "../shuffled_data/shuffled_samples/german_shuffled.jsonl",
            "Spanish": "../shuffled_data/shuffled_samples/spanish_shuffled.jsonl",
            "Hungarian": "../shuffled_data/shuffled_samples/hungarian_shuffled.jsonl"
        }
    }

    # Extract Weight Configurations
    with open("final_test_evaluation_report_detailed.json", 'r') as f:
        weight_configs = json.load(f)
    
    weights_list = [config["weights"] for config in weight_configs]

    # Run Predictions
    predict(path_map["samples"], True, weights_list, poetry_classifier)
    predict(path_map["corpora"], False, weights_list, poetry_classifier)