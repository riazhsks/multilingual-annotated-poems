
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.sparse import hstack, csr_matrix

from common_supervised import load_jsonl_processed, prepare_feature_subsets, save_best_model_and_stats

FORMS_ENG = ['blank verse', 'common measure', 'couplet', 'ballad', 'pantoum',
               'ghazal', 'haiku', 'free verse', 'limerick', 'quatrain', 'tercet',
               'sestina', 'sonnet', 'villanelle']

def load_data(data_file, exclusion_file=None):
    """Loads poem records, filters poems from the hold-out set, and unifies unfixed forms"""
    
    df = load_jsonl_processed(data_file, exclusion_file=None)

    # Handle missing labels
    df = df[df["form"].notnull()].reset_index(drop=True)
    df["form"] = np.where(df["form"].isin(FORMS_ENG), df["form"], "unfixed")

    t_df, n_df, c_df, labels = prepare_feature_subsets(df)
    return t_df, n_df, c_df, labels

def run_ablation_experiments(text_df, numeric_df, categorical_df, labels, train_idx, test_idx, model_type="lr"):
    y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]
    
    # Ablation experiments with textual features
    if  text_df is not None:
        all_features = {
            "text": t_df.columns.tolist(),
            "numeric": n_df.columns.tolist(),
            "categorical": c_df.columns.tolist()
        }
        experiments = [("FULL_MODEL", all_features)]
        for col in all_features["text"]:
            subset = {k: v[:] for k, v in all_features.items()}
            subset["text"].remove(col)
            experiments.append((f"ABLATE_{col}", subset))
        experiments.append(("TEXT_ONLY", {"text": all_features["text"], "numeric": [], "categorical": []}))
        experiments.append(("WITHOUT_TEXT", {"text": [], "numeric":  all_features["numeric"], "categorical": all_features["categorical"]}))
        experiments.append(("WITHOUT_NUMERIC", {"text": all_features["text"], "numeric":  [], "categorical": all_features["categorical"]}))
        experiments.append(("WITHOUT_CATEGORICAL", {"text":all_features["text"], "numeric":  all_features["numeric"], "categorical": []}))

    else:
        all_features = {
            "numeric": n_df.columns.tolist(),
            "categorical": c_df.columns.tolist()
        }
        experiments = [("FULL_MODEL", all_features)]


    # Add an experiment for every numeric feature
    for col in all_features["numeric"]:
        subset = {k: v[:] for k, v in all_features.items()}
        subset["numeric"].remove(col)
        experiments.append((f"ABLATE_{col}", subset))

    # Add an experiment for every categorical feature
    for col in all_features["categorical"]:
        subset = {k: v[:] for k, v in all_features.items()}
        subset["categorical"].remove(col)
        experiments.append((f"ABLATE_{col}", subset))

    experiments.append(("NUMERIC_ONLY", {"text": [], "numeric": all_features["numeric"], "categorical": []}))
    experiments.append(("CATEGORICAL_ONLY", {"text": [], "numeric": [], "categorical": all_features["categorical"]}))

    results_list = []
    per_form_results = []
    best_f1 = -1
    best_bundle = None

    # Run ablation experiments
    for name, feat_set in experiments:

        X_tr_parts, X_te_parts = [], []
        bundle = {
            "name": name, 
            "feature_config": feat_set, 
            "tfidf": {}, 
            "scaler": None, 
            "onehot": None, 
            "model_type": model_type
        }

        # Encode features

        if text_df is not None:
            if feat_set["text"]:
                for col in feat_set["text"]:
                    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(3,5), max_features=1000, min_df=3)
                    X_tr_parts.append(tfidf.fit_transform(text_df.iloc[train_idx][col]))
                    X_te_parts.append(tfidf.transform(text_df.iloc[test_idx][col]))
                    bundle["tfidf"][col] = tfidf

        if feat_set["numeric"]:
            scaler = StandardScaler(with_mean=False)
            X_tr_parts.append(csr_matrix(scaler.fit_transform(numeric_df.iloc[train_idx][feat_set["numeric"]])))
            X_te_parts.append(csr_matrix(scaler.transform(numeric_df.iloc[test_idx][feat_set["numeric"]])))
            bundle["scaler"] = scaler

        if feat_set["categorical"]:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
            X_tr_parts.append(ohe.fit_transform(categorical_df.iloc[train_idx][feat_set["categorical"]]))
            X_te_parts.append(ohe.transform(categorical_df.iloc[test_idx][feat_set["categorical"]]))
            bundle["onehot"] = ohe

        if not X_tr_parts:
            continue
            
        X_train = hstack(X_tr_parts)
        X_test = hstack(X_te_parts)

        # Train LR or SVC model
        if model_type == "lr":
            model = LogisticRegression(max_iter=5000, solver="lbfgs", class_weight="balanced", C=10)
        else:
            model = LinearSVC(class_weight="balanced", C=5, max_iter=10000)

        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_micro = f1_score(y_test, y_pred, average="micro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        report_dict = classification_report(y_test, y_pred, output_dict=True)
        
        res = {
            "Experiment": name, 
            "Accuracy": acc, 
            "Macro_F1": f1_macro, 
            "Micro_F1": f1_micro, 
            "Weighted_F1": f1_weighted
        }
        results_list.append(res)
   
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

        # Track best model
        if f1_macro > best_f1:
            best_f1 = f1_macro
            bundle["model"] = model
            bundle["report"] = report_dict 
            best_bundle = bundle

    return pd.DataFrame(results_list), best_bundle, per_form_results


if __name__ == "__main__":

    DATA_PATH = "../data_processing/english_data/english_poems_processed.jsonl"
    EXCLUSION_PATH = "../shuffled_data/shuffled_samples/english_shuffled.jsonl"
    
    t_df, n_df, c_df, labels = load_data(DATA_PATH, exclusion_file=None)
    
    # Fix a single train-test split for all experiments
    train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.2, random_state=42)
    
    for downsmpl in [True, False]:
        curr_train_idx = train_idx.copy()
        if downsmpl:
            # Logic to downsample sonnets
            tr_lbls = labels.iloc[curr_train_idx]
            sonnet_idx = tr_lbls[tr_lbls == "sonnet"].index.values
            other_idx = tr_lbls[tr_lbls != "sonnet"].index.values
             # Sample 50% of the  sonnets
            np.random.seed(42)
            sampled_sonnets = np.random.choice(sonnet_idx, len(sonnet_idx)//2, replace=False)
            curr_train_idx = np.concatenate([sampled_sonnets, other_idx])

        for delex in [True, False]:
            # Run LR and SVC models for each combination
            for m_type in ["lr", "svc"]:
                mode_dir = "delexicalized" if delex else "full"
                res_df, best_bundle, pf_results = run_ablation_experiments(
                    None if delex else t_df, n_df, c_df, labels, curr_train_idx, test_idx, m_type
                )
                
                sfx = "downsampled" if downsmpl else "standard"
                save_best_model_and_stats(f"english_models_best/{m_type}/{mode_dir}", sfx, res_df, pf_results, best_bundle)