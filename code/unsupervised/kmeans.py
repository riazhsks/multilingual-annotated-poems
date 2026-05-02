import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

# Define paths to annotated poems
base_dir = "../data_processing"
PATHS = {
    "Spanish": f"{base_dir}/spanish_data/spanish_poems_processed.jsonl",
    "English": f"{base_dir}/english_data/english_poems_processed.jsonl",
    "CCV_labeled": f"{base_dir}/czech_data/czech_poems_processed_ccv.jsonl",
    "CCV_unlabeled": f"{base_dir}/czech_data/czech_poems_processed_ccv.jsonl",
    "Hungarian": f"{base_dir}/hungarian_data/hungarian_poems_processed.jsonl",
    "German": f"{base_dir}/german_data/german_poems_processed.jsonl",
    "C3P": f"{base_dir}/czech_data/czech_poems_processed_c3p.jsonl",
}


TEXT_COLS = ["normalized_text", "normalized_title", "normalized_collection", "author"]
NUMERIC_COLS = ["line_count", "total_syllables", "average_syllable_count"]
CATEGORICAL_COLS = ["stanza_scheme", "rhyme_scheme_per_stanza", 
                    "rhyme_scheme", "metrical_foot", "metrical_foot_count"]

FEATURES_TO_ANALYZE = [
    "form", "normalized_title", "author", "normalized_collection", "line_count",
    "stanza_scheme", "rhyme_scheme_per_stanza", "rhyme_scheme", "metrical_foot", 
    "metrical_foot_count", "average_syllable_count"
]

for lang, path in PATHS.items():
    os.makedirs(f"kmeans/{lang}", exist_ok=True)
    
    # Load data
    try:
        df = pd.read_json(path, lines=True)
    except Exception as e:
        print(f"Error loading {lang}: {e}")
        continue
    
    # Handle labeled and unlabeled CCV data separately
    if lang == "CCV_labeled":
        df = df[df["form"].notna()]
    elif lang == "CCV_unlabeled":
        df = df[df["form"].isna()]

    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(0)
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].fillna("").astype(str)
    df[TEXT_COLS] = df[TEXT_COLS].fillna('')


    all_features = {
        "text": TEXT_COLS,
        "numeric": NUMERIC_COLS,
        "categorical": CATEGORICAL_COLS
    }

    # Define ablation experiments
    experiments = []
    for col in all_features["numeric"]:
        subset = {k: v[:] for k, v in all_features.items()}
        subset["numeric"].remove(col)
        experiments.append((f"ABLATE_{col}", subset))
        
    for col in all_features["categorical"]:
        subset = {k: v[:] for k, v in all_features.items()}
        subset["categorical"].remove(col)
        experiments.append((f"ABLATE_{col}", subset))

    lang_results_report = []

    # Run ablation experiments
    for exp_name, subset in experiments:
        os.makedirs(f"kmeans/{lang}/{exp_name}", exist_ok=True)

        transformers = []
        if subset.get("numeric"):
            transformers.append(('num', StandardScaler(), subset["numeric"]))
        if subset.get("categorical"):
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), subset["categorical"]))
        if subset.get("text"):
            for txt_col in subset["text"]:
                max_f = 8000 if txt_col == "normalized_text" else 1000
                transformers.append((f'text_{txt_col}', TfidfVectorizer(max_features=max_f, ngram_range=(1, 2), min_df=2), txt_col))

        # Transform data
        preprocessor = ColumnTransformer(transformers=transformers)
        X_processed = preprocessor.fit_transform(df)

        # Define dimensionality reductions
        reducers = {
            "pca": TruncatedSVD(n_components=2),
            "tsne": TSNE(n_components=2, perplexity=30, init='pca',  learning_rate='auto'),
            "umap": umap.UMAP(n_neighbors=50, min_dist=0.0, init='random', metric='cosine', n_jobs=-1,random_state=None)
        }

        for name, reducer in reducers.items():
            embedding = reducer.fit_transform(X_processed)
            best_k, best_sil, best_labels = 2, -1, None
            
            # Grid search for best k
            for k in range(2, 61): 
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = km.fit_predict(embedding)

                score = silhouette_score(embedding, labels)
                
                if score > best_sil:
                    best_sil, best_k, best_labels = score, k, labels

            df['kmeans_cluster'] = best_labels
            out_path = f"kmeans/{lang}/{exp_name}/best_{name}_k{best_k}"
            os.makedirs(out_path, exist_ok=True)
            

            # Plotting
            plt.figure(figsize=(12, 8))
            sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=df['kmeans_cluster'], palette='tab10', s=80, alpha=0.4)
            plt.title(f"{lang} - {exp_name} ({name.upper()}) Best k={best_k}")
            plt.savefig(f"{out_path}/plot.png")
            plt.savefig(f"{out_path}/plot.pdf")
            plt.close()


            # Cluster summaries
            summary_data = []
            for cid in sorted(np.unique(best_labels)):
                cluster_df = df[df['kmeans_cluster'] == cid]
                cluster_info = {'cluster_id': cid, 'poem_count': len(cluster_df), 'is_noise': (cid == -1)}
                
                # For each feature, find the 3 most frequent values
                for feat in FEATURES_TO_ANALYZE:
                    top_vals = cluster_df[feat].value_counts(normalize=True).head(3)
                    formatted = [f"{val} ({perc:.01%})" for val, perc in top_vals.items()]
                    while len(formatted) < 3: formatted.append("-")
                    cluster_info[f'{feat}_top1'], cluster_info[f'{feat}_top2'], cluster_info[f'{feat}_top3'] = formatted
                summary_data.append(cluster_info)

            pd.DataFrame(summary_data).to_csv(f"{out_path}/analysis.csv", index=False)
            lang_results_report.append({
                "Lang": lang, "Exp": exp_name, "Method": name, 
                "Best_K": best_k, "Sil": best_sil
            })

    results_df = pd.DataFrame(lang_results_report)
    if not results_df.empty:
        results_df = results_df.sort_values(by='Sil', ascending=False)

    results_df.to_csv(f"kmeans/{lang}/{lang}_summary_report.csv", index=False)
