import json
import pandas as pd 

if __name__=="__main__":
    corpora = {
        "CCV": ("czech_data/czech_poems_processed_ccv.jsonl", "czech_data/annotated_ccv.jsonl"),
        "C3P": ("czech_data/czech_poems_processed_c3p.jsonl", "czech_data/standalone_annotated_c3p.jsonl"),
        "German": ("german_data/german_poems_processed.jsonl", "german_data/annotated_german.jsonl"),
        "Hungarian": ("hungarian_data/hungarian_poems_processed.jsonl", "hungarian_data/standalone_annotated_hungarian.jsonl")
    }

    annotated = []
    for corpus, (processed_data, annotated_data) in corpora.items():
        # Load predictions from the most optimal setup
        predictions_file = f"../rule_based/predictions/{corpus}/set5_classification.txt"
        predictions = []
        with open (predictions_file, 'r') as file:
            for line in file:
                prediction = line.split()[0]
                predictions.append(prediction)
        # Enrich data with new poetic form annotations
        with open (annotated_data, "w") as output:
            new_annotations = 0
            with open (processed_data, "r") as data:
                count = 0
                for line in data:
                    poem_record = json.loads(line)
                    new_poem_record = poem_record
                    if poem_record.get("form", None) == None:
                        if predictions[count] != "other":
                            new_poem_record["form"] = predictions[count]
                            new_annotations +=1

                    # Standalone annotations
                    if corpus in ["Hungarian", "C3P"]:
                        del new_poem_record['text']
                        del new_poem_record['normalized_text']
                    json_record = json.dumps(new_poem_record,ensure_ascii=False)
                    output.write(json_record + "\n")     
                    count += 1 
            record = {"processed_data" : corpus, "Annotated" : new_annotations}
            annotated.append(record)
    annotated = pd.DataFrame(annotated)
    annotated.to_csv("new_annotations_stats.csv", index=False)
