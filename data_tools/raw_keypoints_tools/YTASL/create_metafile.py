import h5py
import json
import os
import pdb

file_dir = '/scratch/project_465000977/data/YoutubeASL/features_v2/mae'
#file_path = '/scratch/project_465000977/data/How2Sign/features/keypoints/H2S.keypoints.dev.0.h5'
#file_path = '/scratch/project_465000977/data/YoutubeASL/raw_keypoints/YT_keypoints_raw-0.h5'

metadata = {}
#pdb.set_trace()

for filename in os.listdir(file_dir):
    if filename.startswith("YouTubeASL.mae.train") and filename.endswith(".h5"):
        # Extract the shard ID by splitting the filename
        #shard_id = filename.split('-')[-1].split('.')[0]
        shard_id = filename.split('.')[-2]
        
        # Open the HDF5 file and get the top-level groups
        with h5py.File(os.path.join(file_dir, filename), 'r') as h5file:
            for key in h5file.keys():
                # Map each top-level key to the shard ID
                #core = '_'.join(key.split('_')[:-2])
                core = key.split('.')[0]
                metadata[core] = shard_id

'''
with h5py.File(file_path, 'r') as h5file:
    #h5file.visititems(print_structure)
    for key in h5file.keys():
        print(key)
        #core = '_'.join(key.split('_')[:-2])
        metadata[key] = 0
'''

with open(os.path.join(file_dir, 'YouTubeASL.mae.train.json'), 'w') as f:
    json.dump(metadata, f)

