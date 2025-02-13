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
DEFAULT_OUTPUT_FILE = "evaluation_scores.txt"

def parse_json(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        references = []
        candidates = []
        for video_name, clips in data.items():
            for clip_id, translations in clips.items():
                # skip the 'clip_order' key
                if clip_id == 'clip_order':
                    continue
                references.append(translations['translation'])
                candidates.append(translations['hypothesis'])
        return references, candidates
    except Exception as e:
        print(f"Error parsing JSON file {file_path}: {e}")
        return None, None

def parse_json_files(path):
    try:
        references = []
        candidates = []
        if os.path.isfile(path):
            return parse_json(path)
        elif os.path.isdir(path):
            for filename in os.listdir(path):
                if filename.endswith('.json'):
                    file_refs, file_cands = parse_json(os.path.join(path, filename))
                    if file_refs and file_cands:
                        references.extend(file_refs)
                        candidates.extend(file_cands)
        else:
            raise ValueError(f"Invalid path: {path}")
        return references, candidates
    except Exception as e:
        print(f"Error parsing JSON files: {e}")
        return None, None

def evaluate_with_bleurt(references, candidates, checkpoint):
    try:
        if not tf.io.gfile.exists(checkpoint):
            raise FileNotFoundError(f"BLEURT checkpoint not found at {checkpoint}")
        scorer = score.BleurtScorer(checkpoint)
        scores = scorer.score(references=references, candidates=candidates)
        return scores
    except Exception as e:
        print(f"Error evaluating with BLEURT: {e}")
        return None

def evaluate_with_bleu(references, candidates):
    try:
        bleu = evaluate.load("bleu")
        results = bleu.compute(predictions=candidates, references=[[ref] for ref in references])
        return results['bleu']
    except Exception as e:
        print(f"Error evaluating with BLEU: {e}")
        return None

def evaluate_with_chrf(references, candidates):
    try:
        chrf = evaluate.load("chrf")
        results = chrf.compute(predictions=candidates, references=[[ref] for ref in references])
        return results['score']
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
        # Check if BLEURT checkpoint exists
        if not os.path.exists(args.checkpoint):
            print(f"BLEURT checkpoint not found at {args.checkpoint}")
            print("Attempting to download BLEURT-20 checkpoint...")
            os.system(f"wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip -P {os.path.dirname(args.checkpoint)}")
            os.system(f"unzip {os.path.dirname(args.checkpoint)}/BLEURT-20.zip -d {os.path.dirname(args.checkpoint)}")
            print("BLEURT-20 checkpoint downloaded and extracted.")

        # Parse JSON file(s)
        print("Parsing JSON files...")
        references, candidates = parse_json_files(args.json_path)
        if not references or not candidates:
            raise ValueError("Failed to parse JSON files")
        
        print(f"Number of sentence pairs: {len(references)}")
        
        results = {}
        
        # Evaluate using BLEURT
        if not args.skip_bleurt:
            print("Evaluating with BLEURT...")
            bleurt_scores = evaluate_with_bleurt(references, candidates, args.checkpoint)
            if bleurt_scores:
                results['BLEURT'] = {'scores': bleurt_scores, 'average': mean(bleurt_scores)}
                print(f"BLEURT evaluation complete. Average score: {results['BLEURT']['average']:.4f}")
            else:
                print("BLEURT evaluation failed.")
        
        # Evaluate using BLEU
        if not args.skip_bleu:
            print("Evaluating with BLEU...")
            bleu_score = evaluate_with_bleu(references, candidates)
            if bleu_score is not None:
                results['BLEU'] = bleu_score
                print(f"BLEU evaluation complete. Score: {bleu_score:.4f}")
            else:
                print("BLEU evaluation failed.")
        
        # Evaluate using ChrF
        if not args.skip_chrf:
            print("Evaluating with ChrF...")
            chrf_score = evaluate_with_chrf(references, candidates)
            if chrf_score is not None:
                results['ChrF'] = chrf_score
                print(f"ChrF evaluation complete. Score: {chrf_score:.4f}")
            else:
                print("ChrF evaluation failed.")
        
        # Save the scores to a file
        with open(args.output, 'w') as f:
            f.write(f"Number of evaluated pairs: {len(references)}\n\n")
            if 'BLEURT' in results:
                f.write(f"Average BLEURT score: {results['BLEURT']['average']:.4f}\n")
            if 'BLEU' in results:
                f.write(f"BLEU score (corpus-level): {results['BLEU']:.4f}\n")
            if 'ChrF' in results:
                f.write(f"ChrF score (corpus-level): {results['ChrF']:.4f}\n")
            
            if 'BLEURT' in results:
                f.write("\nIndividual BLEURT scores:\n")
                for i, score in enumerate(results['BLEURT']['scores']):
                    f.write(f"Pair {i+1}: BLEURT={score:.4f}\n")

        print(f"Evaluation results saved to {args.output}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Traceback:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()