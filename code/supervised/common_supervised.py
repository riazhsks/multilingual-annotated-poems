import os
import json
import joblib
import pandas as pd

TEXT_COLS = ["normalized_text", "normalized_title", "normalized_collection", "author"]
NUMERIC_COLS = ["line_count", "total_syllables", "average_syllable_count"]
CATEGORICAL_COLS = ["stanza_scheme", "rhyme_scheme_per_stanza", "rhyme_scheme", "metrical_foot", "metrical_foot_count"]


def load_model(model_file):
    """Loads model and its associated encoders/scalers."""
    bundle = joblib.load(model_file)
    return (
        bundle["model"], 
        bundle.get("scaler"), 
        bundle.get("onehot"), 
        bundle.get("tfidf", {})
    )

def load_jsonl_processed(data_file, exclusion_file=None):
    """Loads JSONL data while filtering out excluded poem IDs."""
    excluded_ids = set()
    if exclusion_file and os.path.exists(exclusion_file):
        with open(exclusion_file, 'r', encoding='utf-8') as f:
            excluded_ids = {line.strip() for line in f if line.strip()}
    
    records = []
    with open(data_file, "r", encoding="utf-8") as inp:
        for line in inp:
            record = json.loads(line)
            if str(record.get("poem_id")) not in excluded_ids:
                records.append(record)
    return pd.DataFrame(records)

def prepare_feature_subsets(df):
    labels = df.get("form")
    t_df = df[TEXT_COLS].fillna("")
    n_df = df[NUMERIC_COLS].fillna(0)
    c_df = df[CATEGORICAL_COLS].fillna("").astype(str)
    return t_df, n_df, c_df, labels

def save_best_model_and_stats(save_dir, suffix, results_df, per_form_results, best_bundle):
    """Save ablation reports and models."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save overall ablation results
    results_df.sort_values(by="Macro_F1", ascending=False).to_csv(
        f"{save_dir}/ablation_{suffix}.csv", index=False
    )
    # Save per-form metrics
    pd.DataFrame(per_form_results).to_csv(
        f"{save_dir}/per_form_metrics_{suffix}.csv", index=False
    )
    # Save best model and detailed report
    pd.DataFrame(best_bundle["report"]).transpose().to_csv(
        f"{save_dir}/best_model_detailed_report_{suffix}.csv"
    )
    joblib.dump(best_bundle, f"{save_dir}/best_model_{suffix}.joblib", compress=3)