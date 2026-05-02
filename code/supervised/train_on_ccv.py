import os
import json
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, accuracy_score
from common_supervised import load_jsonl_processed, prepare_feature_subsets, save_best_model_and_stats

RELIABLE_FORMS = [
    "sonet", "haiku", "limerik", "sonet anglický", 
    "stance", "gazel", "sapfická strofa", "rondel", "tercína", "hrdinský kuplet"
]

def load_data(data_file, exclusion_file=None):
    """Loads poem records, filters poems from the hold-out set, and samples a portion of unlabeled poems"""
    
    df = load_jsonl_processed(data_file, exclusion_file=None)

    # Handle missing labels
    df["form"] = df["form"].fillna("unfixed")
    labels = df["form"]
        
    # Keep all reliable forms + 10% of 'unfixed'
    reliable_df = df[df["form"].isin(RELIABLE_FORMS)]
    unfixed_df = df[df["form"] == "unfixed"]
    unfixed_sampled = unfixed_df.sample(frac=0.1, random_state=42)
    
    df = pd.concat([reliable_df, unfixed_sampled]).reset_index(drop=True)

    t_df, n_df, c_df, labels = prepare_feature_subsets(df)
    return t_df, n_df, c_df, labels

def run_ablation_experiments(t_df, n_df, c_df, labels, train_idx, test_idx, downsample=False):
    """Trains and evaluates models across ablation experiments."""
    y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]
    
    # Define ablation experiments
    all_features = {
        "numeric": n_df.columns.tolist(),
        "categorical": c_df.columns.tolist()
    }
    if t_df is not None:
        all_features["text"] = t_df.columns.tolist()

    experiments = [("FULL_MODEL", all_features)]

    # Ablate individual features
    for group, cols in all_features.items():
        for col in cols:
            subset = {k: v[:] for k, v in all_features.items()}
            subset[group].remove(col)
            experiments.append((f"ABLATE_{col}", subset))

    # Add feature categories combinations experiments
    if t_df is not None:
        experiments.append(("TEXT_ONLY", {"text": all_features["text"], "numeric": [], "categorical": []}))
        experiments.append(("WITHOUT_TEXT", {"text": [], "numeric": all_features["numeric"], "categorical": all_features["categorical"]}))
        experiments.append(("WITHOUT_NUMERIC", {"text": all_features["text"], "numeric": [], "categorical": all_features["categorical"]}))
        experiments.append(("WITHOUT_CATEGORICAL", {"text": all_features["text"], "numeric": all_features["numeric"], "categorical": []}))

    experiments.append(("NUMERIC_ONLY", {"text": [], "numeric": all_features["numeric"], "categorical": []}))
    experiments.append(("CATEGORICAL_ONLY", {"text": [], "numeric": [], "categorical": all_features["categorical"]}))

    results_list = []
    per_form_results = []
    best_f1 = -1
    best_bundle = None

    # Run experiments
    for name, feat_set in experiments:
        X_tr_parts, X_te_parts = [], []
        current_bundle = {
            "name": name,
            "feature_config": feat_set,
            "tfidf": {},
            "scaler": None,
            "onehot": None
        }
        
        # Encode features

        if t_df is not None and feat_set.get("text"):
            for col in feat_set["text"]:
                tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
                X_tr_parts.append(tfidf.fit_transform(t_df.iloc[train_idx][col]))
                X_te_parts.append(tfidf.transform(t_df.iloc[test_idx][col]))
                current_bundle["tfidf"][col] = tfidf

        if feat_set.get("numeric"):
            scaler = StandardScaler(with_mean=False) # with_mean=False for sparse compatibility
            X_tr_parts.append(csr_matrix(scaler.fit_transform(n_df.iloc[train_idx][feat_set["numeric"]])))
            X_te_parts.append(csr_matrix(scaler.transform(n_df.iloc[test_idx][feat_set["numeric"]])))
            current_bundle["scaler"] = scaler

        if feat_set.get("categorical"):
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
            X_tr_parts.append(ohe.fit_transform(c_df.iloc[train_idx][feat_set["categorical"]]))
            X_te_parts.append(ohe.transform(c_df.iloc[test_idx][feat_set["categorical"]]))
            current_bundle["onehot"] = ohe

        # Combine feature matrices
        X_train = hstack(X_tr_parts)
        X_test = hstack(X_te_parts)

        # Train
        model = LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        
        results_list.append({
            "Experiment": name, 
            "Accuracy": acc, 
            "Macro_F1": f1_macro, 
            "Micro_F1": f1_score(y_test, y_pred, average="micro"),
            "Weighted_F1": f1_score(y_test, y_pred, average="weighted")
        })

        for form_name, metrics in report_dict.items():
            if form_name in labels.unique():
                per_form_results.append({
                    "Experiment": name,
                    "Form": form_name,
                    "Precision": metrics['precision'],
                    "Recall": metrics['recall'],
                    "F1-Score": metrics['f1-score'],
                    "Support": metrics['support']
                })

        # Remember the best model
        if f1_macro > best_f1:
            best_f1 = f1_macro
            current_bundle["model"] = model
            current_bundle["report"] = report_dict 
            best_bundle = current_bundle

    # Save results
    dir_type = "full" if t_df is not None else "delexicalized"
    suffix = "downsampled" if downsample else "standard"
    output_dir = f"czech_models_best/{dir_type}"
    results_df = pd.DataFrame(results_list)
    save_best_model_and_stats(output_dir, suffix, results_df, per_form_results, best_bundle)


if __name__ == "__main__":
    DATA_PATH = "../data_processing/czech_data/czech_poems_processed_ccv.jsonl"
    EXCLUSION_PATH = "../shuffled_data/shuffled_samples/czech_ccv_shuffled.jsonl"
    
    t_df, n_df, c_df, labels = load_data(DATA_PATH, exclusion_file=EXCLUSION_PATH)

    # Get a single train-test split for all experiments
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    

    for downsmpl in [True, False]:
        current_train_idx = train_idx.copy()
        
        # Sample 5% of sonnets
        if downsmpl:
            train_labels = labels.iloc[current_train_idx]
            is_sonnet = train_labels.str.contains("sonet", case=False)
            
            sonnet_indices = train_labels[is_sonnet].index.values
            other_indices = train_labels[~is_sonnet].index.values
            
            np.random.seed(42)
            sonnet_sample_size = max(1, int(len(sonnet_indices) * 0.05))
            sampled_sonnets = np.random.choice(sonnet_indices, size=sonnet_sample_size, replace=False)
            current_train_idx = np.concatenate([sampled_sonnets, other_indices])
        
        # Run full and delexicalized models
        for delex in [True, False]:
            text_data = None if delex else t_df
            run_ablation_experiments(
                text_data, n_df, c_df, labels, current_train_idx, test_idx, downsample=downsmpl
            )
