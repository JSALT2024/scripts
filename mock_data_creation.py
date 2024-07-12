import h5py
import json
import numpy as np
import os
import string
import random


def generate_random_text(length):
    vocabulary = "The job's resource requirements are specified, indicating the \
    necessary resources for the job to execute on the compute nodes. \
    These specifications include the job name, output filename, RAM capacity, \
    number of CPUs, nodes, tasks, time constraints, and other relevant parameters. \
    These commands, known as SBATCH directives, must be written in uppercase format \
    and preceded by a pound sign.".split()
    random_text = " ".join(random.sample(vocabulary, random.randint(5, 20)))
    return random_text


if __name__ == '__main__':
    output_path = './h5py'

    json_dict = {}
    metadata_mae = {}
    metadata_dino = {}
    metadata_sign2vec = {}
    metadata_pose = {}

    number_of_videos = 14
    video_names = [f'video_{i}' for i in range(1, number_of_videos + 1)]
    max_videos_per_group = 5
    number_of_groups = int(number_of_videos / max_videos_per_group)
    mae_files = {}
    dino_files = {}
    sign2vec_files = {}
    pose_files = {}
    for group_id in range(number_of_groups+1):
        mae_files[f'mae_{group_id}'] = h5py.File(os.path.join(output_path, f'mae_{group_id}.h5'), 'w')
        dino_files[f'dino_{group_id}'] = h5py.File(os.path.join(output_path, f'dino_{group_id}.h5'), 'w')
        sign2vec_files[f'sign2vec_{group_id}'] = h5py.File(os.path.join(output_path, f'sign2vec_{group_id}.h5'), 'w')
        pose_files[f'pose_{group_id}'] = h5py.File(os.path.join(output_path, f'pose_{group_id}.h5'), 'w')
    for video_id, video in enumerate(video_names):
        current_group = int(video_id / max_videos_per_group)
        metadata_mae[video] = current_group
        metadata_dino[video] = current_group
        metadata_sign2vec[video] = current_group
        metadata_pose[video] = current_group
        num_of_clips = np.random.randint(1, 15)
        clip_names = [f'clip_{i}' for i in range(1, num_of_clips + 1)]
        mae_h5 = mae_files[f'mae_{current_group}'].create_group(video)
        dino_h5 = dino_files[f'dino_{current_group}'].create_group(video)
        sign2vec_h5 = sign2vec_files[f'sign2vec_{current_group}'].create_group(video)
        pose_h5 = pose_files[f'pose_{current_group}'].create_group(video)
        json_dict[video] = {"clip_order": []}
        for clip in clip_names:
            json_dict[video]["clip_order"].append(clip)
            text_length = np.random.randint(5, 20)
            json_dict[video][clip] = {"translation": generate_random_text(text_length),
                                      "paraphrases": [generate_random_text(text_length) for _ in range(3)]}

            num_of_frames = np.random.randint(5, 30)

            features_mae = np.random.rand(num_of_frames, 768)
            mae_h5.create_dataset(clip, shape=(features_mae.shape[0], features_mae.shape[1]), dtype=np.float16)
            mae_h5[clip][:] = features_mae

            features_dino = np.random.rand(num_of_frames, 384)
            dino_h5.create_dataset(clip, shape=(features_dino.shape[0], features_dino.shape[1]), dtype=np.float16)
            dino_h5[clip][:] = features_dino

            features_sign2vec = np.random.rand(num_of_frames, 768)
            sign2vec_h5.create_dataset(clip, shape=(features_sign2vec.shape[0], features_sign2vec.shape[1]),
                                       dtype=np.float16)
            sign2vec_h5[clip][:] = features_sign2vec

            features_pose = np.random.rand(num_of_frames, 32)
            pose_h5.create_dataset(clip, shape=(features_pose.shape[0], features_pose.shape[1]), dtype=np.float16)
            pose_h5[clip][:] = features_pose

    for file in mae_files.values():
        file.close()
    for file in dino_files.values():
        file.close()
    for file in sign2vec_files.values():
        file.close()
    for file in pose_files.values():
        file.close()

    with open(os.path.join(output_path, 'annotation.json'), 'w') as f:
        json.dump(json_dict, f)

    with open(os.path.join(output_path, 'metadata_mae.json'), 'w') as f:
        json.dump(metadata_mae, f)

    with open(os.path.join(output_path, 'metadata_dino.json'), 'w') as f:
        json.dump(metadata_dino, f)

    with open(os.path.join(output_path, 'metadata_sign2vec.json'), 'w') as f:
        json.dump(metadata_sign2vec, f)

    with open(os.path.join(output_path, 'metadata_pose.json'), 'w') as f:
        json.dump(metadata_pose, f)
