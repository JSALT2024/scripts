import json
import pandas as pd
from tqdm import tqdm

json_file = '../data/final_jsons/h2s.annotations.test.new.json'

with open(json_file, 'r') as f:
    data = json.load(f)

df = pd.read_csv('../data/rephrase/how2sign_realigned_test_rephrased_4o-mini.csv', sep="\t", quotechar='\x07')

for video in tqdm(data):
    for clip in data[video]['clip_order']:
        data[video][clip]['paraphrases'] = []
        result = df[(df['SENTENCE_NAME'] == clip)].iloc[0]
        if result['GPT_Rephrase1'] == '<none>':
            continue
        else:
            data[video][clip]['paraphrases'].append(result['GPT_Rephrase1'])
            data[video][clip]['paraphrases'].append(result['GPT_Rephrase2'])
            data[video][clip]['paraphrases'].append(result['GPT_Rephrase3'])
            data[video][clip]['paraphrases'].append(result['GPT_Rephrase4'])
            data[video][clip]['paraphrases'].append(result['GPT_Rephrase5'])

with open(json_file[:-8]+'gpt4.json', 'w') as f:
    json.dump(data, f, indent=4)
