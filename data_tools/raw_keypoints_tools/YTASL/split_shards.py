import json
import h5py
import pdb
import os

# pdb.set_trace()

# Load the JSON file and extract the keys (IDs)
json_file_path = '/scratch/project_465000977/data/YoutubeASL/features/mae/16-07_21-52-12/YouTubeASL.mae.dev.json'
with open(json_file_path, 'r') as f:
    json_data = json.load(f)
    json_keys = list(json_data.keys())

# Open the original H5 file for reading
# h5_file_path = '/scratch/project_465000977/data/YoutubeASL/features_v2/keypoints/yt.keypoints.all.0.h5'
# dev_h5_file_path = '/scratch/project_465000977/data/YoutubeASL/features_v2/keypoints/yt.keypoints.dev.0.h5'
# train_h5_file_path = '/scratch/project_465000977/data/YoutubeASL/features_v2/keypoints/yt.keypoints.train.0.h5'

input_h5_file_dir = '/scratch/project_465000977/data/YoutubeASL/features_v2/mae'

for filename in os.listdir(os.path.join(input_h5_file_dir, 'all')):
    if filename.startswith("yt.mae.train") and filename.endswith(".h5"):
        shard_id = filename.split('.')[-2]

        with h5py.File(os.path.join(os.path.join(input_h5_file_dir, 'all'), filename), 'r') as all_h5, \
             h5py.File(os.path.join(input_h5_file_dir, 'YouTubeASL.mae.dev.{}.h5'.format(shard_id)), 'w') as dev_h5, \
             h5py.File(os.path.join(input_h5_file_dir, 'YouTubeASL.mae.train.{}.h5'.format(shard_id)), 'w') as train_h5:

            all_h5_keys = list(all_h5.keys())
            #pdb.set_trace()
            # Iterate over each key in the H5 file
            for key in all_h5_keys:
                # Extract the ID portion from the key
                #video_id = key.split('.')[0]  # Assumes key format: '{id}.{timestamp}'
                video_id = key

                # Check if this ID exists in the JSON keys
                if video_id in json_keys:
                    # Copy the data to the 'dev' H5 file
                    all_h5.copy(key, dev_h5)
                else:
                    # Copy the data to the 'other' H5 file
                    all_h5.copy(key, train_h5)

