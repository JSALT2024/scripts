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
                references.append(translations['ref'])
                candidates.append(translations['output'])
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
        return results
    except Exception as e:
        print(f"Error evaluating with BLEU: {e}")
        return None


def evaluate_with_chrf(references, candidates):
    try:
        chrf = evaluate.load("chrf")
        results = chrf.compute(predictions=candidates, references=[[ref] for ref in references])
        return results
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
            bleu_results = evaluate_with_bleu(references, candidates)
            if bleu_results:
                results['BLEU'] = bleu_results
                print(f"BLEU evaluation complete. Score: {bleu_results['bleu']:.4f}")
            else:
                print("BLEU evaluation failed.")
        
        # Evaluate using ChrF
        if not args.skip_chrf:
            print("Evaluating with ChrF...")
            chrf_results = evaluate_with_chrf(references, candidates)
            if chrf_results:
                results['ChrF'] = chrf_results
                print(f"ChrF evaluation complete. Score: {chrf_results['score']:.4f}")
            else:
                print("ChrF evaluation failed.")
        
        # Save the scores to a file
        with open(args.output, 'w') as f:
            f.write(f"Number of evaluated pairs: {len(references)}\n\n")
            for metric, score in results.items():
                if metric == 'BLEURT':
                    f.write(f"Average BLEURT score: {score['average']:.4f}\n")
                elif metric == 'BLEU':
                    f.write(f"BLEU score: {score['bleu']:.4f}\n")
                elif metric == 'ChrF':
                    f.write(f"ChrF score: {score['score']:.4f}\n")
            
            f.write("\nIndividual scores:\n")
            for i in range(len(references)):
                f.write(f"Pair {i+1}:")
                if 'BLEURT' in results:
                    f.write(f" BLEURT={results['BLEURT']['scores'][i]:.4f}")
                if 'BLEU' in results:
                    if 'precisions' in results['BLEU'] and len(results['BLEU']['precisions']) > i:
                        f.write(f" BLEU={results['BLEU']['precisions'][i]:.4f}")
                    else:
                        f.write(f" BLEU=N/A")
                if 'ChrF' in results:
                    f.write(f" ChrF={results['ChrF']['score']:.4f}")
                f.write("\n")
        print(f"Evaluation results saved to {args.output}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Traceback:")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()