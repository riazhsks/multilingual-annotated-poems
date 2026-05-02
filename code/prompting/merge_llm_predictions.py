import json
import pandas as pd


FILES = {"../shuffled_data/shuffled_samples/english_shuffled.jsonl" : "english_predictions.txt",
         "../shuffled_data/shuffled_samples/spanish_shuffled.jsonl" : "spanish_predictions.txt",
         "../shuffled_data/shuffled_samples/czech_c3p_shuffled.jsonl" : "czech_c3p_predictions.txt",
         "../shuffled_data/shuffled_samples/czech_ccv_shuffled.jsonl" : "czech_ccv_predictions.txt",
         "../shuffled_data/shuffled_samples/german_shuffled.jsonl" : "german_predictions.txt",
         "../shuffled_data/shuffled_samples/hungarian_shuffled.jsonl" : "hungarian_predictions.txt"}


PREDICTIONS_DIRS = ["gemini-2.5-pro/P1", "gemini-2.5-pro/P2", "gpt-4o/P1", "gpt-4o/P2"]

results = []
all_dfs = []

for data, predictions in FILES.items():

    # Define the corpus name
    if "c3p" in predictions:
        corpus = "C3P"
    elif "ccv" in predictions:
        corpus = "CCV"
    else:
        corpus = predictions.split("_")[0].capitalize()

    # Load ground truth predictions
    ground_truth_records = []
    line_counter = 1
    
    with open(data, "r", encoding="utf-8") as gt_file:
        for json_line in gt_file:
            poem_object = json.loads(json_line)
            true_form = (poem_object.get("form", "") or "").lower()
            
            ground_truth_records.append({
                "Poem ID": line_counter, 
                "Corpus": corpus, 
                "Ground Truth": true_form
            })
            line_counter += 1

    df_corpus = pd.DataFrame(ground_truth_records)

    for directory in PREDICTIONS_DIRS:
        # Determine specific model and prompt ID from the directory path
        model_name, prompt_id = directory.split("/")

        if "gemini" in model_name:
            model_name = "Gemini"
        else:
            model_name = "GPT"
        column_header = f"{model_name} {prompt_id}"
        
        prediction_records = []
        prediction_line_id = 1
        
        # Read the LLM output file
        prediction_filepath = f"{directory}/{predictions}"
        with open(prediction_filepath, "r", encoding="utf-8") as pred_file:
            raw_content = pred_file.read()
            
            # Handle messy JSON formatting the models created
            json_array_string = "[" + raw_content.replace("}{", "},{").replace("}\n{", "},{") + "]"
            predicted_poems = json.loads(json_array_string)
            
            for poem_entry in predicted_poems:
                predicted_form = (poem_entry.get("form", "") or "").lower()
                prediction_records.append({
                    "Poem ID": prediction_line_id, 
                    "Corpus": corpus, 
                    column_header: predicted_form
                })
                prediction_line_id += 1
        
        # Merge the current LLM's column into the corpus dataframe
        prediction_df = pd.DataFrame(prediction_records)
        df_corpus = pd.merge(
            df_corpus, 
            prediction_df, 
            on=["Poem ID", "Corpus"], 
            how="left"
        )

    all_dfs.append(df_corpus)

# Combine all corpora into one final table
final_df = pd.concat(all_dfs, ignore_index=True)
final_df.to_csv("llm_preds.csv", index=False)
