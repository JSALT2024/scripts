import json
import numpy as np

with open(r'C:\Work\JSALT\scripts\data\yt.annotations.train.new.json', 'r') as f:
    annotations = json.load(f)

for video in annotations:
    video_data = annotations[video]
    for clip in video_data['clip_order']:
        if str(video_data[clip]['translation']) == 'nan':
            video_data['clip_order'].remove(clip)
            del video_data[clip]
            print(f'{video}/{clip} removed')

with open(r'C:\Work\JSALT\scripts\data\yt.annotations.train.new.json', 'w') as f:
    json.dump(annotations, f, indent=4)
