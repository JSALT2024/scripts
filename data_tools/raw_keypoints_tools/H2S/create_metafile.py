import h5py
import json

file_path = 'H2S_test.h5'
#file_path = '/scratch/project_465000977/data/How2Sign/features/keypoints/H2S.keypoints.dev.0.h5'
#file_path = '/scratch/project_465000977/data/YoutubeASL/raw_keypoints/YT_keypoints_raw-0.h5'

def print_structure(name, obj):
    print(name)

metadata = {}

with h5py.File(file_path, 'r') as h5file:
    #h5file.visititems(print_structure)
    for key in h5file.keys():
        print(key)
        #core = '_'.join(key.split('_')[:-2])
        metadata[key] = 0

with open('metadata_test.json', 'w') as f:
    json.dump(metadata, f)
