import pandas as pd

# Load predictions
eng = pd.read_csv("full_eng_models_preds.csv")
czech = pd.read_csv("full_cz_models_preds.csv")

# Merge
combined_df = pd.concat([eng, czech], ignore_index=True)

# Save the result
combined_df.to_csv("full_preds.csv", index=False)
