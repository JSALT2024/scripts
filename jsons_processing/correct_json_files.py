import json

with open('data/yt.annotations-nodup.train.json', 'r') as f:
    annotations = json.load(f)

for video in annotations:
    video_data = annotations[video]
    for clip in video_data['clip_order']:

        if video_data[clip]['paraphrases'][0] == '<none>':
            video_data[clip]['paraphrases'] = []

with open('data/yt.annotations-nodup.train.new.json', 'w') as f:
    json.dump(annotations, f, indent=4)
