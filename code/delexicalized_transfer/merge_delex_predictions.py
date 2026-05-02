import pandas as pd

# Load predictions
eng = pd.read_csv("delex_eng_models_preds.csv")
czech = pd.read_csv("delex_cz_models_preds.csv")

# Merge
merged = pd.merge(czech, eng, on=["Corpus", "Poem ID"], suffixes=('', ' eng'))

if 'Ground Truth eng' in merged.columns:
    merged = merged.drop(columns=['Ground Truth eng'])

final_comparison = merged.groupby(['Corpus', 'Poem ID', 'Ground Truth'], dropna=False).first().reset_index()

final_comparison.to_csv("delexicalized_preds.csv", index=False)

