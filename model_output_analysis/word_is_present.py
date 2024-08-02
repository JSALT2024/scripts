import json
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def parse_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    all_true = []
    all_pred = []
    
    for video_id, video_data in data.items():
        for clip_id, clip_data in video_data.items():
            if clip_id != "clip_order" and isinstance(clip_data, dict):
                if 'reference' in clip_data and 'hypothesis' in clip_data:
                    ref_answers = clip_data['reference'].lower().split(',')
                    hyp_answers = clip_data['hypothesis'].lower().split(',')
                    
                    # Ensure both lists have the same length
                    min_length = min(len(ref_answers), len(hyp_answers))
                    ref_answers = ref_answers[:min_length]
                    hyp_answers = hyp_answers[:min_length]
                    
                    y_true = [ans.strip() == 'yes' for ans in ref_answers]
                    y_pred = [ans.strip() == 'yes' for ans in hyp_answers]
                    
                    all_true.extend(y_true)
                    all_pred.extend(y_pred)
    
    return all_true, all_pred

def evaluate_classification(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate word presence detection")
    parser.add_argument("json_path", help="Path to JSON file")
    parser.add_argument("--output", default="word_presence_results.txt", help="Output file for evaluation results")
    args = parser.parse_args()

    try:
        # Parse JSON file
        print("Parsing JSON file...")
        y_true, y_pred = parse_json(args.json_path)
        
        if not y_true or not y_pred:
            raise ValueError("Failed to parse JSON file or no valid data found")
        
        print(f"Number of evaluated answers: {len(y_true)}")
        
        # Evaluate classification
        results = evaluate_classification(y_true, y_pred)
        
        # Print results
        print("\nWord Presence Detection Results:")
        for metric, value in results.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        # Save results to file
        with open(args.output, 'w') as f:
            f.write(f"Number of evaluated answers: {len(y_true)}\n\n")
            f.write("Word Presence Detection Results:\n")
            for metric, value in results.items():
                f.write(f"{metric.capitalize()}: {value:.4f}\n")
        
        print(f"\nResults saved to {args.output}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()