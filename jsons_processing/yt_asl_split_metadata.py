import json
from tqdm import tqdm

with open(r'C:\Work\JSALT\scripts\data\ytasl\YT.annotations.train.json') as f:
    train_data = json.load(f)

with open(r'C:\Work\JSALT\scripts\data\ytasl\YT.annotations.val.json') as f:
    val_data = json.load(f)

dino_json = r'C:\Work\JSALT\scripts\data\ytasl\YouTubeASL.dino.train.old.json'
mae_json = r'C:\Work\JSALT\scripts\data\ytasl\YouTubeASL.mae.train.old.json'
keypoints_json = r'C:\Work\JSALT\scripts\data\ytasl\YouTubeASL.keypoints.train.old.json'


with open(dino_json) as f:
    dino_data = json.load(f)

with open(mae_json) as f:
    mae_data = json.load(f)

with open(keypoints_json) as f:
    keypoints_data = json.load(f)

dino_train = {}
dino_val = {}

mae_train = {}
mae_val = {}

keypoints_train = {}
keypoints_val = {}


for vid in tqdm(dino_data):
    if vid in train_data:
        dino_train[vid] = dino_data[vid]
    elif vid in val_data:
        dino_val[vid] = dino_data[vid]


with open(r'C:\Work\JSALT\scripts\data\ytasl\YouTubeASL.dino.train.json', 'w') as f:
    json.dump(dino_train, f, indent=4)

with open(r'C:\Work\JSALT\scripts\data\ytasl\YouTubeASL.dino.val.json', 'w') as f:
    json.dump(dino_val, f, indent=4)


for vid in tqdm(mae_data):
    if vid in train_data:
        mae_train[vid] = mae_data[vid]
    elif vid in val_data:
        mae_val[vid] = mae_data[vid]

with open(r'C:\Work\JSALT\scripts\data\ytasl\YouTubeASL.mae.train.json', 'w') as f:
    json.dump(mae_train, f, indent=4)

with open(r'C:\Work\JSALT\scripts\data\ytasl\YouTubeASL.mae.val.json', 'w') as f:
    json.dump(mae_val, f, indent=4)

for vid in tqdm(keypoints_data):
    if vid in train_data:
        keypoints_train[vid] = keypoints_data[vid]
    elif vid in val_data:
        keypoints_val[vid] = keypoints_data[vid]

with open(r'C:\Work\JSALT\scripts\data\ytasl\YouTubeASL.keypoints.train.json', 'w') as f:
    json.dump(keypoints_train, f, indent=4)

with open(r'C:\Work\JSALT\scripts\data\ytasl\YouTubeASL.keypoints.val.json', 'w') as f:
    json.dump(keypoints_val, f, indent=4)
