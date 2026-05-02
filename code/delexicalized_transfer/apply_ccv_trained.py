import json
import joblib
import pandas as pd
import os
from scipy.sparse import hstack, csr_matrix
from ..supervised.apply_ccv_trained import load_data, apply_model
from ..supervised.common_supervised import  load_model


if __name__ == "__main__":
    DATA = ["code/shuffled_data/shuffled_samples/czech_c3p_shuffled.jsonl", "code/shuffled_data/shuffled_samples/czech_ccv_shuffled.jsonl",
            "code/shuffled_data/shuffled_samples/hungarian_shuffled.jsonl", "code/shuffled_data/shuffled_samples/german_shuffled.jsonl",
            "code/shuffled_data/shuffled_samples/english_shuffled.jsonl", "code/shuffled_data/shuffled_samples/spanish_shuffled.jsonl"]
    
    BASE_MODEL_PATH = "code/supervised/czech_models_best"
    all_results = []

    for data in DATA:
        if "ccv" in data:
            corpus = "CCV"
        elif "c3p" in data:
            corpus = "C3P"
        else:
            corpus = data.split("/")[-1].split("_")[0].capitalize()

        # Load data once per file
        _, n_df, c_df, labels = load_data(data)

        for sampling in ["standard", "downsampled"]:
            
            # Construct the specific config ID (Czech prefix)
            config_id = f"Cz-d-{sampling[0]}"
            
            # Map to the folder structure
            model_rel_path = f"delexicalized/best_model_{sampling}.joblib"
            full_path = os.path.join(BASE_MODEL_PATH, model_rel_path)
            
            # Load the bundle
            model, scaler, onehot, tfidf = load_model(full_path)

            current_text_df = None
            
            # Apply and save
            df_res = apply_model(corpus, config_id, current_text_df, n_df, c_df, labels, 
                                 model, scaler, onehot, tfidf)
            all_results.append(df_res)

    if all_results:
        master_df = pd.concat(all_results, ignore_index=True)
        master_df = master_df.groupby(['Corpus', 'Poem ID', 'Ground Truth'], dropna=False).first().reset_index()
        master_df.to_csv("code/delexicalized_transfer/delex_cz_models_preds.csv", index=False)
