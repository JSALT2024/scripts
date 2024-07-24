import json
import argparse
from pprint import pprint

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def print_all_data(data):
    print("All Translation Pairs:")
    for video_name, clips in data.items():
        for clip_id, translations in clips.items():
            print(f"\nVideo: {video_name}")
            print(f"Clip ID: {clip_id}")
            print(f"Reference: {translations['ref']}")
            print(f"Output: {translations['output']}")

def main():
    parser = argparse.ArgumentParser(description="Pretty print all JSON data")
    parser.add_argument("json_path", help="Path to JSON file")
    args = parser.parse_args()

    try:
        data = load_json(args.json_path)
        print_all_data(data)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()