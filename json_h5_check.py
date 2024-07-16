import json
import h5py

data_dir = './data'

data_modality = 'pose'

with open(f'{data_dir}/annotations.dev.new.json', 'r') as f:
    annotations = json.load(f)

with open(f'{data_dir}/pose/metadata_pose.dev.json', 'r') as f:
    metadata = json.load(f)

video_names = list(annotations.keys())

total_number_of_founded_clips = 0
total_number_of_missing_clips = 0

new_annotations = {}

for video_name in video_names:
    correct_h5_file = metadata[video_name]
    clip_order = annotations[video_name]['clip_order']

    h5py_file = h5py.File(f'{data_dir}/{data_modality}/{data_modality}.dev.{correct_h5_file}.h5', 'r')
    for clip in clip_order:
        try:
            features = h5py_file[video_name][clip]
            if features is not None:
                total_number_of_founded_clips += 1
                if video_name not in new_annotations:
                    new_annotations[video_name] = {'clip_order': []}
                new_annotations[video_name]['clip_order'].append(clip)
                new_annotations[video_name][clip] = annotations[video_name][clip]
                continue
            else:
                print(f'{video_name}/{clip} is None!!!!!!')
        except KeyError:
            total_number_of_missing_clips += 1
            print(f'{video_name}/{clip} does not exist!!!!!!')

print(f'Total number of founded clips: {total_number_of_founded_clips}')
print(f'Total number of missing clips: {total_number_of_missing_clips}')

with open(f'{data_dir}/annotations.dev.new.json', 'w') as f:
    json.dump(new_annotations, f, indent=4)
