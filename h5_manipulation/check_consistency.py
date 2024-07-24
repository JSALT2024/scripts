import json
import h5py
from tqdm import tqdm

with open(r'C:\Work\JSALT\scripts\data\pose\h2s.keypoints.train.json', 'r') as f:
    pose_ann = json.load(f)

with open(r'C:\Work\JSALT\scripts\data\sign2vec\H2S.sign2vec'
          r'.train.json', 'r') as f:
    dino_ann = json.load(f)

for video in pose_ann:
    if video not in dino_ann:
        print(f'{video} not in dino')
        continue

list_of_h5s = [r'C:\Work\JSALT\scripts\data\sign2vec\H2S.sign2vec.train.0.h5',]
# list_of_h5s =  [r'C:\Work\JSALT\scripts\data\mae\H2S.mae.train.0.h5',
#                 r'C:\Work\JSALT\scripts\data\mae\H2S.mae.train.1.h5',
#                 r'C:\Work\JSALT\scripts\data\mae\H2S.mae.train.2.h5',
#                 r'C:\Work\JSALT\scripts\data\mae\H2S.mae.train.3.h5',
#                 r'C:\Work\JSALT\scripts\data\mae\H2S.mae.train.4.h5',
#                 r'C:\Work\JSALT\scripts\data\mae\H2S.mae.train.5.h5',
#                 r'C:\Work\JSALT\scripts\data\mae\H2S.mae.train.6.h5',
#                 r'C:\Work\JSALT\scripts\data\mae\H2S.mae.train.7.h5',
#                 r'C:\Work\JSALT\scripts\data\mae\H2S.mae.train.8.h5',
#                 r'C:\Work\JSALT\scripts\data\mae\H2S.mae.train.9.h5']

with h5py.File(r'C:\Work\JSALT\scripts\data\pose\pose.train.0.h5','r') as fr:
    keys = list(fr.keys())

    clips_num = 0
    for key in tqdm(keys):
        correct_h5 = list_of_h5s[dino_ann[key]]
        with h5py.File(correct_h5, 'r') as fd:
            if key not in fd:
                print(f'{key} not in {correct_h5}')
                continue
            for sub_key in fr[key].keys():
                if sub_key not in fd[key]:
                    print(f'{sub_key} not in {correct_h5}/{key}')
                    continue
                clips_num += 1

print(clips_num)
