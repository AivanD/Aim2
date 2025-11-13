import pandas as pd
import json
import os
from sklearn.metrics import confusion_matrix, classification_report
import logging

from aim2.utils.logging_cfg import setup_logging
from aim2.utils.config import PROCESSED_RE_OUTPUT_DIR, LOGS_DIR, RE_OUTPUT_DIR

# --- Configuration ---
GOLD_STANDARD_FILE = os.path.join(RE_OUTPUT_DIR, 'annotated/GS_relations.xlsx')
PREDICTIONS_DIR = PROCESSED_RE_OUTPUT_DIR
LOG_FILE = os.path.join(LOGS_DIR, 'evaluation.log')

def load_predictions(pmcid):
    """Loads the predicted relations for a given PMCID."""
    prediction_file = os.path.join(PREDICTIONS_DIR, f"{pmcid}.json")
    if not os.path.exists(prediction_file):
        logging.warning(f"Prediction file not found for PMCID: {pmcid}")
        return None
    try:
        with open(prediction_file, 'r') as f:
            data = json.load(f)
            # The file contains a dictionary with a 'relations' key
            return data.get('relations', [])
    except json.JSONDecodeError:
        logging.error(f"Could not decode JSON from {prediction_file}")
        return None
    except Exception as e:
        logging.error(f"Error reading {prediction_file}: {e}")
        return None

def main():
    """
    Main function to load data, perform comparison, and generate evaluation metrics.
    """
    setup_logging(log_file_name="evaluation.log")
    logging.info("Starting relation extraction evaluation...")

    # 1. Load Gold Standard data
    try:
        gs_df_raw = pd.read_excel(GOLD_STANDARD_FILE)
        # Define the columns we need and how to rename them
        column_mapping = {
            'PMCID': 'pmcid',
            'Subject_name': 'subject_name',
            'Object_name': 'object_name',
            'Object Alt Names': 'object_alt_names',
            'Actual Relation (GPT-Thinking + Manual)': 'actual_relation'
        }
        
        # Select and rename the required columns
        required_cols = [col for col in column_mapping.keys() if col in gs_df_raw.columns]
        gs_df = gs_df_raw[required_cols].rename(columns=column_mapping)

        # Ensure alt names column exists and fill NaNs
        if 'object_alt_names' not in gs_df.columns:
            gs_df['object_alt_names'] = ''
        gs_df['object_alt_names'] = gs_df['object_alt_names'].fillna('')

        logging.info(f"Successfully loaded {len(gs_df)} gold standard relations.")
    except FileNotFoundError:
        logging.error(f"Gold standard file not found at: {GOLD_STANDARD_FILE}")
        return
    except Exception as e:
        logging.error(f"Error reading Excel file: {e}")
        return

    y_true = []
    y_pred = []
    
    # Create a set of all predicted pairs for easy lookup of False Positives
    all_predicted_pairs = set()

    # Group by PMCID to process one paper at a time
    for pmcid, group in gs_df.groupby('pmcid'):
        logging.info(f"Processing PMCID: {pmcid}...")
        predictions = load_predictions(pmcid)
        if predictions is None:
            # If no prediction file, all GS relations for this PMCID are False Negatives
            for _, row in group.iterrows():
                y_true.append(row['actual_relation'])
                y_pred.append('No_Relationship') # Model failed to predict anything
            continue

        # Create a dictionary for quick lookup of predictions by (subject, object)
        pred_dict = {
            (p['subject_entity']['name'], p['object_entity']['name']): p['predicate']
            for p in predictions
        }
        
        # Store all predicted pairs for this PMCID
        for pair in pred_dict.keys():
            all_predicted_pairs.add((pmcid, pair[0], pair[1]))

        # 2. Compare Gold Standard with Predictions (TP, FN, FP-class)
        for _, row in group.iterrows():
            subject_name = row['subject_name']
            actual_relation = row['actual_relation']
            
            # Create a set of all possible valid object names
            main_object_name = row['object_name']
            alt_object_names_str = row.get('object_alt_names', '')
            
            possible_object_names = {main_object_name}
            if alt_object_names_str:
                # Assuming alt names are separated by comma and optional space
                alt_names = [name.strip() for name in alt_object_names_str.split(';')]
                possible_object_names.update(alt_names)

            y_true.append(actual_relation)
            
            predicted_relation = None
            # Check if any of the possible pairs exist in the predictions
            for obj_name in possible_object_names:
                if (subject_name, obj_name) in pred_dict:
                    predicted_relation = pred_dict[(subject_name, obj_name)]
                    break # Found a match, stop checking other alt names
            
            if predicted_relation is not None:
                # Pair was found in predictions (TP or FP-class)
                y_pred.append(predicted_relation)
                # Log a warning if the prediction does not match the gold standard
                if predicted_relation != actual_relation:
                    logging.warning(f"FP (Classification) for {pmcid}: Pair ({subject_name}, {main_object_name}) - GS says '{actual_relation}', model predicted '{predicted_relation}'.")
            else:
                # Pair was not found in predictions (FN or TN)
                y_pred.append('No_Relationship')
                # Only log a warning if a real relationship was missed (it's a true FN)
                if actual_relation != 'No_Relationship':
                    logging.warning(f"FN for {pmcid}: Pair ({subject_name}, {main_object_name}) with relation '{actual_relation}' not found in predictions.")

    # 3. Identify False Positives (FP-existence)
    # These are pairs predicted by the model but not present in the gold standard
    gs_pairs_set = set()
    for _, row in gs_df.iterrows():
        pmcid = row['pmcid']
        subject_name = row['subject_name']
        main_object_name = row['object_name']
        gs_pairs_set.add((pmcid, subject_name, main_object_name))
        
        alt_object_names_str = row.get('object_alt_names', '')
        if alt_object_names_str:
            alt_names = [name.strip() for name in alt_object_names_str.split(';')]
            for alt_name in alt_names:
                if alt_name: # Avoid adding empty strings
                    gs_pairs_set.add((pmcid, subject_name, alt_name))

    for pmcid, subject, object_ in all_predicted_pairs:
        if (pmcid, subject, object_) not in gs_pairs_set:
            # This is a False Positive
            y_true.append('No_Relationship') # The true relation is non-existent
            
            # We need to fetch the prediction again for this pair
            predictions = load_predictions(pmcid)
            if predictions:
                pred_dict = {
                    (p['subject_entity']['name'], p['object_entity']['name']): p['predicate']
                    for p in predictions
                }
                predicted_relation = pred_dict.get((subject, object_), 'No_Relationship')
                y_pred.append(predicted_relation)
                logging.warning(f"FP for {pmcid}: Pair ({subject}, {object_}) predicted as '{predicted_relation}' but not in Gold Standard.")


    # 4. Calculate and display metrics
    if not y_true or not y_pred:
        logging.error("No data to evaluate. Both y_true and y_pred are empty.")
        return

    # Get all unique labels from both true and predicted sets
    labels = sorted(list(set(y_true) | set(y_pred)))

    logging.info("\n" + "="*50 + "\nEvaluation Results\n" + "="*50)

    # Confusion Matrix
    try:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        
        logging.info("\nConfusion Matrix:\n" + cm_df.to_string())

        # Per-class metrics
        for i, label in enumerate(labels):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - (tp + fn + fp)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            logging.info(f"Class '{label}': TP={tp}, FP={fp}, FN={fn}, TN={tn}, Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1:.4f}")

    except Exception as e:
        logging.error(f"Could not generate confusion matrix: {e}")

    # Classification Report
    try:
        report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
        logging.info("\nClassification Report:\n" + report)
    except Exception as e:
        logging.error(f"Could not generate classification report: {e}")

    logging.info("Evaluation finished.")


if __name__ == "__main__":
    main()