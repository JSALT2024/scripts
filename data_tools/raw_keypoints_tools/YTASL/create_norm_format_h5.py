import json
import h5py
import os
import pdb
import numpy as np

input_dir = '/scratch/project_465000977/data/YoutubeASL/raw_keypoints_new/raw'
output_dir = '/scratch/project_465000977/data/YoutubeASL/raw_keypoints_new/norm_format'

face_indexes = [0, 4, 13, 14, 17, 33, 39, 46, 52, 55, 61, 64, 81, 93, 133, 151, 152, 159, 172, 178, 181, 263, 269, 276, 282, 285, 291, 294, 311, 323, 362, 386, 397, 402, 405, 468, 473 ]

for filename in os.listdir(input_dir):
    filepath = os.path.join(input_dir, filename)
    if not filepath.endswith('.h5'):
        continue
    print(f'Processing {filepath}')

    shard_id = filename.split('-')[-1].split('.')[0]
    split_name = filename.split('.')[2]
    new_filename = 'YouTubeASL.keypoints.{}.{}.h5'.format(split_name, shard_id)

    outputfilepath = os.path.join(output_dir, new_filename)

    with h5py.File(filepath, 'r') as input_h5, \
         h5py.File(outputfilepath, 'w') as output_h5:

        # Iterate over each key in the H5 file
        for key in input_h5.keys():
            # Extract the ID portion from the key
            video_id, timestamp = key.split('.')  # Assumes key format: '{id}.{timestamp}'

            #pdb.set_trace()

            pose_landmarks = input_h5[key]['joints']['pose_landmarks'][()]
            face_landmarks = input_h5[key]['joints']['face_landmarks'][()]
            left_hand_landmarks = input_h5[key]['joints']['left_hand_landmarks'][()]
            right_hand_landmarks = input_h5[key]['joints']['right_hand_landmarks'][()]

            pose_landmarks = pose_landmarks[:, :25, :2]
            face_landmarks = face_landmarks[:, face_indexes, :2]
            left_hand_landmarks = left_hand_landmarks[:, :, :2]
            right_hand_landmarks = right_hand_landmarks[:, :, :2]

            data = np.concatenate((pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks), axis=1).reshape(len(face_landmarks), 208)

            if video_id not in output_h5:
                video_id_group = output_h5.create_group(video_id)
            else:
                video_id_group = output_h5[video_id]

            video_id_group.create_dataset(key, data=data)

print('Dataset created')
