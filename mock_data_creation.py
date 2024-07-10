import h5py
import json
import numpy as np
import os

dt = h5py.vlen_dtype(np.dtype('float64'))

output_path = './h5py'
mae = h5py.File(os.path.join(output_path, 'mae.h5'), 'w')
dino = h5py.File(os.path.join(output_path, 'dino.h5'), 'w')
sign2vec = h5py.File(os.path.join(output_path, 'sign2vec.h5'), 'w')
pose = h5py.File(os.path.join(output_path, 'pose.h5'), 'w')

json_dict = {}

video_names = ['video1', 'video2', 'video3', 'video4', 'video5']
for video in video_names:
    num_of_clips = np.random.randint(1, 15)
    clip_names = [f'{i}' for i in range(1, num_of_clips + 1)]
    mae_h5 = mae.create_group(video)
    dino_h5 = dino.create_group(video)
    sign2vec_h5 = sign2vec.create_group(video)
    pose_h5 = pose.create_group(video)

    json_dict[video] = clip_names
    for clip in clip_names:
        num_of_frames = np.random.randint(5, 30)

        features_mae = np.random.rand(num_of_frames, 768)
        mae_h5.create_dataset(clip, shape=(features_mae.shape[0], features_mae.shape[1]), dtype=np.float32)
        mae_h5[clip][:] = features_mae

        features_dino = np.random.rand(num_of_frames, 384)
        dino_h5.create_dataset(clip, shape=(features_dino.shape[0], features_dino.shape[1]), dtype=np.float32)
        dino_h5[clip][:] = features_dino

        features_sign2vec = np.random.rand(num_of_frames, 768)
        sign2vec_h5.create_dataset(clip, shape=(features_sign2vec.shape[0], features_sign2vec.shape[1]), dtype=np.float32)
        sign2vec_h5[clip][:] = features_sign2vec

        features_pose = np.random.rand(num_of_frames, 32)
        pose_h5.create_dataset(clip, shape=(features_pose.shape[0], features_pose.shape[1]), dtype=np.float32)
        pose_h5[clip][:] = features_pose

mae.close()
dino.close()
sign2vec.close()
pose.close()