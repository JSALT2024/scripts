import json
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def parse_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    y_true = []
    y_pred = []
    
    for video_id, video_data in data.items():
        for clip_id, clip_data in video_data.items():
            if clip_id != "clip_order" and isinstance(clip_data, dict):
                if 'reference' in clip_data and 'hypothesis' in clip_data:
                    # Only include pairs where both reference and hypothesis are non-empty
                    if clip_data['reference'] and clip_data['hypothesis']:
                        y_true.append(clip_data['reference'].lower().strip() == 'yes')
                        y_pred.append(clip_data['hypothesis'].lower().strip() == 'yes')
    
    return y_true, y_pred

def evaluate_binary_classification(y_true, y_pred):
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
    parser = argparse.ArgumentParser(description="Evaluate binary classification for reversed order detection")
    parser.add_argument("json_path", help="Path to JSON file")
    parser.add_argument("--output", default="binary_classification_results.txt", help="Output file for evaluation results")
    args = parser.parse_args()

    try:
        # Parse JSON file
        print("Parsing JSON file...")
        y_true, y_pred = parse_json(args.json_path)
        
        if not y_true or not y_pred:
            raise ValueError("Failed to parse JSON file or no valid data found")
        
        print(f"Number of evaluated pairs: {len(y_true)}")
        
        # Evaluate binary classification
        results = evaluate_binary_classification(y_true, y_pred)
        
        # Print results
        print("\nBinary Classification Results:")
        for metric, value in results.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        # Save results to file
        with open(args.output, 'w') as f:
            f.write(f"Number of evaluated pairs: {len(y_true)}\n\n")
            f.write("Binary Classification Results:\n")
            for metric, value in results.items():
                f.write(f"{metric.capitalize()}: {value:.4f}\n")
        
        print(f"\nResults saved to {args.output}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()