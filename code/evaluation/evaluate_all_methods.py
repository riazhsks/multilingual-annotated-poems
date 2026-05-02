
import pandas as pd

# Define all prediction labels 
RuleBasedSet = ["sonnet",'sonet', 'sonet anglický','limerik','gazel',"limerick","ghazal","sestina","couplet", 'hrdinský kuplet']

Prompt2Set = ['sonnet', 'couplet', 'quatrain', 'ghazal', 'haiku', 'limerick', 'sestina', 'villanelle', 'blank verse', 'common measure', 
              'rondel', 'knittelvers', 'liedstrophe', 'sonett', 'dal', 'bokorrím', 'felező tizenkettes', 'romance', 'copla', 'décima', 
              'redondilla', 'byliny', 'chastushka', 'dolnik', 'sapfická strofa', 'tercína', 'hrdinský kuplet', 'stance']

CzechEngMap = {'sonet':'sonnet','sonet anglický':'sonnet', 'limerik':'limerick', 'gazel':'ghazal', 'hrdinský kuplet':'couplet'}

FixedForms = ['sonet','sonnet','sonet anglický','limerik','limerick', 'gazel','ghazal', 'haiku','rondel', 
              'ballad', 'pantoum', 'sestina', 'villanelle', 'tercína']

FormalElements = ['hrdinský kuplet','couplet', 'stance', 'sapfická strofa', 'blank verse', 'common measure', 
                  'free verse', 'quatrain', 'tercet']

UnfixedForms = ['ode', 'pastoral', 'aubade', 'dramatic monologue', 'ekphrasis', 'elegy', 'concrete or pattern poetry', 
                'prose poem', 'ars poetica', 'madrigal', 'unfixed']

FixedOrFormal = FixedForms + FormalElements

SupervisedCzech = ['sonet','sonnet', 'limerick', 'ghazal','couplet','limerik', 'sonet anglický', 'gazel', 'rondel', 
                   'tercína', 'stance', 'sapfická strofa', 'hrdinský kuplet']

SupervisedEng = ['ballad','sonet', 'sonet anglický','pantoum', 'limerik','gazel','ghazal','hrdinský kuplet', 'haiku', 'limerick', 
                 'sestina', 'sonnet', 'villanelle', 'blank verse', 'common measure', 'couplet', 'free verse', 'quatrain', 'tercet']

# Evaluate a prediction given ground truth and source of the prediction
def evaluate_prediction(row, col_name):
    # Get the prediction
    prediction = str(row.get(col_name, "")).lower().strip()

    # N/A method (e.g. not delexicalized cross-lingual)
    if prediction in ["nan", "none", ""]:
        return None

    # Get ground-trurth (annotated in the source corpus, or manually)
    # Prefer source over manual (e.g. madrigal for CCV was not identified as any form by the human expert)
    truth_vals = row.get('Ground Truth')
    if pd.isna(truth_vals):
        truth_vals = row.get('Manual Annotation')

    # Detect the experiment
    is_rule_based = col_name.startswith("S")
    is_sl_cz = col_name.startswith("Cz")
    is_sl_eng = col_name.startswith("Eng")
    is_p2 = "P2" in col_name
    
    # Go over both options for cases like 'unfixed/quatrain' 
    for truth in truth_vals.split("/"):
        # Trivial string or substring match (e.g. couplet and heroic couplet)
        if truth in prediction or prediction in truth:
            return "+"
        
        # Checks for cross-lingual models
        if CzechEngMap.get(prediction) == truth or CzechEngMap.get(truth) == prediction:
            return "+"
        
        # Handle the special 'other' class for the rule-based models
        if is_rule_based:
            if truth not in RuleBasedSet:
                return "+" if prediction == "other" else "-"
            if prediction == "other":
                    return "-"


        if is_sl_cz:
            # Something completely unseen
            if truth not in SupervisedCzech:
                return "+" if prediction == "unfixed" else "-" 
        if is_sl_eng:
            # Something completely unseen
            if truth not in SupervisedEng:
                return "+" if prediction == "unfixed" else "-" 
            
            
        # Prompting with P1 ommitted because same as General checks + manual eval
                        
        # Prompting with Prompt2Set
        if is_p2:
            if truth in Prompt2Set:
                # Incorrectly missed the prediction
                if prediction not in truth and truth not in prediction: return "-"
            # Truth is NOT in the restricted set 
            else:
                return "+" if prediction == "unfixed" else "-"

        # General checks

        # If Truth is Unfixed
        if truth in UnfixedForms:
            if prediction in FixedOrFormal:
                return "-"
            if prediction == "unfixed":
                return "+"
            
        # If Truth is a specific fixed form
        if truth in FixedOrFormal:
            if prediction in UnfixedForms:
                return "-"
            if prediction in FixedOrFormal:
                # If they reached here, it's a mismatch (e.g., sonnet vs limerick)
                return "-"

    # Return any prediction that should be checked manually
    return prediction 


# Load the data
data = pd.read_csv("all_predictions.csv")

processed_data = data.copy()

# Extract columns relevant for evaluation
exclude_cols = [ 'Poem ID', 'Corpus', 'Manual Annotation', 'Ground Truth']
prediction_cols = [c for c in processed_data.columns if c not in exclude_cols]

# Evaluate each prediction
for col in prediction_cols:
    processed_data[col] = processed_data.apply(lambda row: evaluate_prediction(row, col), axis=1)

# Save the results
processed_data.to_csv("all_predictions_evaluated.csv", index=False)