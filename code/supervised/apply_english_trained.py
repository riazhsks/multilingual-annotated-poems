import json
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix
import os
from .common_supervised import load_jsonl_processed, prepare_feature_subsets, load_model


def load_data(data_file):
    """Loads poem records, filters poems from the hold-out set, and unifies unfixed forms"""
    
    df = load_jsonl_processed(data_file)

    labels = df["form"] 
    df = df.drop(columns=["form"])

    t_df, n_df, c_df, labels = prepare_feature_subsets(df)
    return t_df, n_df, c_df, labels

def apply_model(lang_name, suffix, text_df, numeric_df, categorical_df, labels, model, scaler, onehot, tfidf):
    """Transforms features, predicts, and saves statistical reports."""
    X_parts = []

    if text_df is not None and tfidf:
        for col, vectorizer in tfidf.items():
            if col in text_df.columns:
                X_parts.append(vectorizer.transform(text_df[col]))

    if scaler:
        expected_num = scaler.feature_names_in_
        X_parts.append(csr_matrix(scaler.transform(numeric_df[expected_num])))

    if onehot:
        expected_cat = onehot.feature_names_in_
        for col in expected_cat:
            if col not in categorical_df.columns:
                categorical_df[col] = ""
        X_parts.append(onehot.transform(categorical_df[expected_cat]))

    X = hstack(X_parts)
    y_pred = model.predict(X)  

    return pd.DataFrame({
        "Corpus": lang_name,
        "Poem ID": range(1, len(y_pred) + 1),
        "Ground Truth": labels.values if labels is not None else None,
        suffix: y_pred,
    })

if __name__ == "__main__":
    DATA = ["code/shuffled_data/shuffled_samples/english_shuffled.jsonl"]
    
    BASE_MODEL_PATH = "code/supervised/english_models_best"
    all_results = []

    for data in DATA:
        # Load data once per file
        t_df, n_df, c_df, labels = load_data(data)

        for m_type in ["lr", "svc"]:
            for sampling in ["standard", "downsampled"]:

                config_id = f"Eng-{m_type.upper()}-f-{sampling[0]}"
                
                model_rel_path = f"{m_type}/full/best_model_{sampling}.joblib"
                full_path = os.path.join(BASE_MODEL_PATH, model_rel_path)
                
                model, scaler, onehot, tfidf = load_model(full_path)
                
                current_text_df = t_df
                
                # Apply and save
                df_res = apply_model("English", config_id, current_text_df, n_df, c_df, labels, 
                                     model, scaler, onehot, tfidf)
                all_results.append(df_res)

    if all_results:
        master_df = pd.concat(all_results, ignore_index=True)
        master_df = master_df.groupby(['Corpus', 'Poem ID', 'Ground Truth'], dropna=False).first().reset_index()
        master_df.to_csv("code/supervised/full_eng_models_preds.csv", index=False)