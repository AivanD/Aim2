import json
import pandas as pd
from collections import defaultdict
import os

def calculate_and_print_metrics(ground_truth, predictions, label="Overall"):
    """Calculates and prints precision, recall, and F1-score."""
    true_positives = len(ground_truth.intersection(predictions))
    false_positives = len(predictions - ground_truth)
    false_negatives = len(ground_truth - predictions)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"--- {label} Performance ---")
    print(f"True Positives:  {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-Score:  {f1_score:.4f}\n")

    return {"precision": precision, "recall": recall, "f1_score": f1_score}

def write_details_to_file(output_path, details):
    """Writes the detailed TP, FP, FN for each label to a text file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("NER Evaluation Details\n")
        f.write("="*40 + "\n\n")
        for label, data in sorted(details.items()):
            f.write(f"--- {label} ---\n")
            
            # False Negatives (in ground truth, but not predicted)
            f.write(f"\n[False Negatives ({len(data['fn'])}): Entities missed by the pipeline]\n")
            for item in sorted(list(data['fn'])):
                f.write(f"  - Entity: '{item[1]}', Type: {item[0]}, Span: ({item[2]}, {item[3]})\n")
                
            # False Positives (predicted, but not in ground truth)
            f.write(f"\n[False Positives ({len(data['fp'])}): Entities incorrectly predicted by the pipeline]\n")
            for item in sorted(list(data['fp'])):
                f.write(f"  - Entity: '{item[1]}', Type: {item[0]}, Span: ({item[2]}, {item[3]})\n")

            # True Positives (in both)
            f.write(f"\n[True Positives ({len(data['tp'])}): Entities correctly predicted]\n")
            for item in sorted(list(data['tp'])):
                f.write(f"  - Entity: '{item[1]}', Type: {item[0]}, Span: ({item[2]}, {item[3]})\n")
                
            f.write("\n" + "="*40 + "\n\n")
    print(f"Detailed evaluation report saved to: {output_path}")

def evaluate_ner(ground_truth_path, prediction_path):
    """
    Evaluates NER performance against a ground truth file, both overall and per-label.
    """
    # 1. Load ground truth data
    try:
        gt_df = pd.read_excel(ground_truth_path)
        # Ensure columns are correct types
        gt_df['entity'] = gt_df['entity'].astype(str)
        gt_df['start'] = pd.to_numeric(gt_df['start'], errors='coerce')
        gt_df['end'] = pd.to_numeric(gt_df['end'], errors='coerce')
        gt_df.dropna(subset=['start', 'end'], inplace=True)

    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {ground_truth_path}")
        return
    except Exception as e:
        print(f"Error reading or processing ground truth file {ground_truth_path}: {e}")
        return

    gt_entities = set(
        gt_df.apply(lambda row: (row['entity type'], row['entity'].lower(), int(row['start']), int(row['end'])), axis=1)
    )

    # 2. Load and flatten predicted data
    try:
        with open(prediction_path, 'r') as f:
            predicted_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Prediction file not found at {prediction_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {prediction_path}: {e}")
        return

    predicted_entities = set()
    for passage in predicted_data:
        for entity_type, entities in passage.items():
            if not entities:
                continue
            for entity in entities:
                if 'name' not in entity or not entity['name']:
                    continue
                for span in entity.get("spans", []):
                    predicted_entities.add((entity_type, entity['name'].lower(), span[0], span[1]))

    # --- Overall Evaluation ---
    all_metrics = {"overall": calculate_and_print_metrics(gt_entities, predicted_entities)}
    print("="*40 + "\n")

    # --- Per-Label Evaluation ---
    all_entity_types = sorted(list(set(row[0] for row in gt_entities) | set(row[0] for row in predicted_entities)))
    
    all_metrics["per_label"] = {}
    evaluation_details = {}
    for entity_type in all_entity_types:
        gt_subset = {e for e in gt_entities if e[0] == entity_type}
        pred_subset = {e for e in predicted_entities if e[0] == entity_type}
        
        if not gt_subset and not pred_subset:
            continue

        metrics = calculate_and_print_metrics(gt_subset, pred_subset, label=entity_type)
        all_metrics["per_label"][entity_type] = metrics

        # Store details for file output
        evaluation_details[entity_type] = {
            'tp': gt_subset.intersection(pred_subset),
            'fp': pred_subset - gt_subset,
            'fn': gt_subset - pred_subset
        }

    # --- Write detailed report to a file ---
    output_dir = os.path.dirname(prediction_path)
    details_filename = os.path.splitext(os.path.basename(prediction_path))[0] + "_evaluation_details.txt"
    details_filepath = os.path.join(output_dir, details_filename)
    write_details_to_file(details_filepath, evaluation_details)

    return all_metrics

if __name__ == '__main__':
    # Example usage:
    # Assumes you have 'PMC7384185.xlsx' in your data directory
    # and the processed output in your output/processed directory.
    gt_file = 'output/ner/annotated/PMC7384185.xlsx' 
    pred_file = 'output/ner/processed/PMC7384185_llama4.json'
    evaluate_ner(gt_file, pred_file)