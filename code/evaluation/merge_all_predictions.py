
import pandas as pd
import os
import numpy as np

# manual annotations of 30 poems per ccv and c3p
manual_c3p = pd.read_csv("../manual_annotations_c3p.csv")
manual_ccv = pd.read_csv("../manual_annotations_ccv.csv")
manual_all = pd.concat([manual_c3p, manual_ccv], ignore_index=True)

# predictions of full SL models
full = pd.read_csv("../supervised/full_preds.csv")

# predictions of delexicalized SL models
delex = pd.read_csv("../delexicalized_transfer/delexicalized_preds.csv")
delex = delex[delex['Corpus'].isin(["English","Spanish","C3P","CCV"])]

# predictions of rule-based models
rb = pd.read_csv("../rule_based/rule_based_preds.csv")
rb = rb[rb['Corpus'].isin(["English","Spanish","C3P","CCV"])]

# predictions of llms
prompting = pd.read_csv("../prompting/llm_preds.csv")
prompting =prompting[prompting['Corpus'].isin(["English","Spanish","C3P","CCV"])]


# merge all predictions and truths

merged_df = prompting.copy()

merged_df = merged_df.merge(manual_all[['Poem ID', 'Corpus', 'Manual Annotation']], on=['Poem ID', 'Corpus'], how='left')
merged_df = merged_df.merge(full, on=['Poem ID', 'Corpus'], how='left', suffixes=('', ' full'))
merged_df = merged_df.merge(delex, on=['Poem ID', 'Corpus'], how='left', suffixes=('', ' delex'))
merged_df = merged_df.merge(rb, on=['Poem ID', 'Corpus'], how='left', suffixes=('', ' rb'))

if 'Ground Truth full' in merged_df.columns:
    merged_df['Ground Truth'] = merged_df['Ground Truth'].fillna(merged_df['Ground Truth full'])
if 'Ground Truth delex' in merged_df.columns:
    merged_df['Ground Truth'] = merged_df['Ground Truth'].fillna(merged_df['Ground Truth delex'])
if 'Ground Truth rb' in merged_df.columns:
    merged_df['Ground Truth'] = merged_df['Ground Truth'].fillna(merged_df['Ground Truth rb'])


cols_to_drop = [c for c in merged_df.columns if ' delex' in c or ' rb' in c]
merged_df = merged_df.drop(columns=cols_to_drop)

# Define final column order 
final_cols = ['Poem ID', 'Corpus', 'Manual Annotation', 'Ground Truth', 'Gemini P1', 'Gemini P2', 'GPT P1', 'GPT P2', 
              'Cz-d-d', 'Cz-d-s','Cz-f-d', 'Cz-f-s', 'Eng-LR-d-d','Eng-LR-d-s','Eng-SVC-d-d','Eng-SVC-d-s',
              'Eng-LR-f-d','Eng-LR-f-s','Eng-SVC-f-d','Eng-SVC-f-s', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']

merged_df = merged_df[final_cols]

merged_df.to_csv("all_predictions.csv", index=False)