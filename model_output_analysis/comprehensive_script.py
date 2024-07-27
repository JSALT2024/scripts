import json
import os
import sys
import argparse
import tensorflow as tf
import evaluate
from statistics import mean
from bleurt import score

# Default paths
DEFAULT_CHECKPOINT_PATH = os.path.expanduser("~/BLEURT-20")
DEFAULT_OUTPUT_FILE = "comprehensive_evaluation.txt"

def parse_json(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        parsed_data = []
        for video_name, clips in data.items():
            for clip_id, translations in clips.items():
                parsed_data.append({
                    'video_name': video_name,
                    'clip_id': clip_id,
                    'reference': translations['ref'],
                    'candidate': translations['output']
                })
        return parsed_data
    except Exception as e:
        print(f"Error parsing JSON file {file_path}: {e}")
        return None

def parse_json_files(path):
    try:
        parsed_data = []
        if os.path.isfile(path):
            return parse_json(path)
        elif os.path.isdir(path):
            for filename in os.listdir(path):
                if filename.endswith('.json'):
                    file_data = parse_json(os.path.join(path, filename))
                    if file_data:
                        parsed_data.extend(file_data)
        else:
            raise ValueError(f"Invalid path: {path}")
        return parsed_data
    except Exception as e:
        print(f"Error parsing JSON files: {e}")
        return None

def evaluate_with_bleurt(data, checkpoint):
    try:
        if not tf.io.gfile.exists(checkpoint):
            raise FileNotFoundError(f"BLEURT checkpoint not found at {checkpoint}")
        scorer = score.BleurtScorer(checkpoint)
        references = [item['reference'] for item in data]
        candidates = [item['candidate'] for item in data]
        scores = scorer.score(references=references, candidates=candidates)
        return scores
    except Exception as e:
        print(f"Error evaluating with BLEURT: {e}")
        return None

def evaluate_with_bleu(data):
    try:
        bleu = evaluate.load("bleu")
        scores = []
        for item in data:
            result = bleu.compute(predictions=[item['candidate']], references=[[item['reference']]])
            scores.append(result['bleu'])
        return scores
    except Exception as e:
        print(f"Error evaluating with BLEU: {e}")
        return None

def evaluate_with_chrf(data):
    try:
        chrf = evaluate.load("chrf")
        scores = []
        for item in data:
            result = chrf.compute(predictions=[item['candidate']], references=[[item['reference']]])
            scores.append(result['score'])
        return scores
    except Exception as e:
        print(f"Error evaluating with ChrF: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate translations using BLEURT, BLEU, and ChrF")
    parser.add_argument("json_path", help="Path to JSON file or directory containing JSON files")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH, help="Path to BLEURT checkpoint")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="Output file for evaluation scores")
    parser.add_argument("--skip-bleurt", action="store_true", help="Skip BLEURT evaluation")
    parser.add_argument("--skip-bleu", action="store_true", help="Skip BLEU evaluation")
    parser.add_argument("--skip-chrf", action="store_true", help="Skip ChrF evaluation")
    args = parser.parse_args()

    try:
        # Parse JSON file(s)
        print("Parsing JSON files...")
        data = parse_json_files(args.json_path)
        if not data:
            raise ValueError("Failed to parse JSON files")
        
        print(f"Number of sentence pairs: {len(data)}")
        
        # Evaluate using BLEURT
        if not args.skip_bleurt:
            print("Evaluating with BLEURT...")
            bleurt_scores = evaluate_with_bleurt(data, args.checkpoint)
            if bleurt_scores:
                for item, score in zip(data, bleurt_scores):
                    item['BLEURT'] = score
                print(f"BLEURT evaluation complete. Average score: {mean(bleurt_scores):.4f}")
            else:
                print("BLEURT evaluation failed.")
        
        # Evaluate using BLEU
        if not args.skip_bleu:
            print("Evaluating with BLEU...")
            bleu_scores = evaluate_with_bleu(data)
            if bleu_scores:
                for item, score in zip(data, bleu_scores):
                    item['BLEU'] = score
                print(f"BLEU evaluation complete. Average score: {mean(bleu_scores):.4f}")
            else:
                print("BLEU evaluation failed.")
        
        # Evaluate using ChrF
        if not args.skip_chrf:
            print("Evaluating with ChrF...")
            chrf_scores = evaluate_with_chrf(data)
            if chrf_scores:
                for item, score in zip(data, chrf_scores):
                    item['ChrF'] = score
                print(f"ChrF evaluation complete. Average score: {mean(chrf_scores):.4f}")
            else:
                print("ChrF evaluation failed.")
        
        # Save the comprehensive results to a file
        with open(args.output, 'w') as f:
            f.write(f"Number of evaluated pairs: {len(data)}\n\n")
            
            # Write average scores
            f.write("Average Scores:\n")
            if 'BLEURT' in data[0]:
                f.write(f"BLEURT: {mean([item['BLEURT'] for item in data]):.4f}\n")
            if 'BLEU' in data[0]:
                f.write(f"BLEU: {mean([item['BLEU'] for item in data]):.4f}\n")
            if 'ChrF' in data[0]:
                f.write(f"ChrF: {mean([item['ChrF'] for item in data]):.4f}\n")
            f.write("\n")

            # Write individual results
            for i, item in enumerate(data, 1):
                f.write(f"Pair {i}:\n")
                f.write(f"Video: {item['video_name']}, Clip ID: {item['clip_id']}\n")
                f.write(f"Reference: {item['reference']}\n")
                f.write(f"Candidate: {item['candidate']}\n")
                f.write("Scores: ")
                if 'BLEURT' in item:
                    f.write(f"BLEURT={item['BLEURT']:.4f} ")
                if 'BLEU' in item:
                    f.write(f"BLEU={item['BLEU']:.4f} ")
                if 'ChrF' in item:
                    f.write(f"ChrF={item['ChrF']:.4f}")
                f.write("\n\n")

        print(f"Comprehensive evaluation results saved to {args.output}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Traceback:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()