import json
from natsort import natsorted

with open(r'C:\Work\JSALT\scripts\data\final_jsons\H2S.annotations.train.gpt4.json') as f:
    data = json.load(f)

for video in data:
    clip_order = data[video]["clip_order"]
    new_clip_order = natsorted(clip_order)
    data[video]["clip_order"] = new_clip_order

with open(r'C:\Work\JSALT\scripts\data\final_jsons\H2S.annotations.train.json', 'w') as f:
    json.dump(data, f, indent=4)
