import json
from tqdm import tqdm

json_file = 'data/train.filtered3.beg_dur_id_frames_fps_text.norm.filter-lanid.tsv+ok-kw8b.json'
output_file = 'data/yt_keywords_dict.txt'

with open(json_file, 'r') as f:
    data = json.load(f)

keywords_dict = []
for video_name in tqdm(data.keys()):
    for clip in data[video_name]['clip_order']:
        keywords = data[video_name][clip]['keywords']
        for keyword in keywords:
            if keyword not in keywords_dict:
                keywords_dict.append(keyword)

with open(output_file, 'w', encoding='utf8') as f:
    for keyword in keywords_dict:
        f.write(keyword + '\n')
