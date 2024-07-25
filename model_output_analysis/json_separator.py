import json
import argparse
import os

def parse_json(file_path):
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

def write_to_file(data, file_path, sentence_type):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(f"{item['video_name']}\t{item['clip_id']}\t{item[sentence_type]}\n")

def write_to_tsv(data, file_path, sentence_type):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("video_name\tclip_id\tsentence\n")  # Header
        for item in data:
            f.write(f"{item['video_name']}\t{item['clip_id']}\t{item[sentence_type]}\n")

def main():
    parser = argparse.ArgumentParser(description="Separate JSON into reference and candidate sentences")
    parser.add_argument("json_path", help="Path to JSON file")
    parser.add_argument("output_dir", nargs="?", default=".", help="Directory to save output files")
    parser.add_argument("--output_dir", dest="output_dir_flag", help="Directory to save output files (alternative way to specify)")
    args = parser.parse_args()

    # Use the flag version if provided, otherwise use the positional argument
    output_dir = args.output_dir_flag if args.output_dir_flag else args.output_dir

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Parse JSON file
        data = parse_json(args.json_path)

        # Write references
        write_to_file(data, os.path.join(output_dir, "references.txt"), "reference")
        write_to_tsv(data, os.path.join(output_dir, "references.tsv"), "reference")

        # Write candidates
        write_to_file(data, os.path.join(output_dir, "candidates.txt"), "candidate")
        write_to_tsv(data, os.path.join(output_dir, "candidates.tsv"), "candidate")

        print(f"Files have been created in {output_dir}:")
        print("- references.txt")
        print("- references.tsv")
        print("- candidates.txt")
        print("- candidates.tsv")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
  