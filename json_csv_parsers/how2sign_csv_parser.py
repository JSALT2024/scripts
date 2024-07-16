import csv
import json

def parse(csv_dir):
    data = {}
    try:
        with open(csv_dir, mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file, delimiter='\t')
            for row in csv_reader:
                video_id = row['VIDEO_ID']
                clip_id = row['SENTENCE_ID'].split('_')[-1]
                caption = row['SENTENCE']

                if video_id not in data:
                    data[video_id] = {}

                data[video_id][clip_id] = {
                    "translation": caption,
                    "paraphrases": [row['GPT_Rephrase1'], row['GPT_Rephrase2'], row['GPT_Rephrase3'], row['GPT_Rephrase4'], row['GPT_Rephrase5']],
                    "keywords": row['KEYWORDS'].strip('\''),
                }
    except UnicodeDecodeError:
        with open(csv_dir, mode='r', encoding='latin1') as file:
            csv_reader = csv.DictReader(file, delimiter='\t')
            for row in csv_reader:
                video_id = row['VIDEO_ID']
                clip_id = row['SENTENCE_ID']
                caption = row['SENTENCE']

                if video_id not in data:
                    data[video_id] = {}

                data[video_id][clip_id] = {
                    "translation": caption,
                    "paraphrases": [row['GPT_Rephrase1'], row['GPT_Rephrase2'], row['GPT_Rephrase3'], row['GPT_Rephrase4'], row['GPT_Rephrase5']],
                    "keywords": row['KEYWORDS'].strip('\'')
                }
    return data

def add_clip_id_list(data):
    new_data = {}
    
    for key, value in data.items():
        clip_id_list = [sub_key for sub_key in value.keys()]
        new_data[key] = {"clip_order": clip_id_list}
        new_data[key].update(value)
        
    return new_data

def write_json(data, output_name):
    with open(output_name, 'w') as file:
        json.dump(data, file, indent=4)

def main(input_dir, output_dir):
    data_json = parse(input_dir)
    updated_json_data = add_clip_id_list(data_json)
    write_json(updated_json_data, output_dir)

if __name__ == '__main__':
    input_dir = 'csv_parsers/h2sign/how2sign_realigned_train.norm.h2s+ok-kw.tsv'
    output_dir = 'csv_parsers/h2sign/annotation.train.json'
    
    main(input_dir, output_dir)