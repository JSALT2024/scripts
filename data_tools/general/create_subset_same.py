import h5py
import json
import numpy as np
import os
import pdb

data_path="/scratch/project_465000977/data/How2Sign/features"

# Load the shard dictionary and annotation file
with open(os.path.join(data_path, 'keypoints/H2S.keypoints.train.json'), 'r') as f:
    shard_dict = json.load(f)

with open(os.path.join(data_path, 'H2S.annotations.train.json'), 'r') as f:
    annotations = json.load(f)

# Create dictionaries for new annotation and metadata files
new_annotations = {}
new_metadata = {}

# Create a list to store selected clips and their metadata
selected_clips = []

max_len = 128

pdb.set_trace()

# Open HDF5 shard 0 (adjust the path if needed)
with h5py.File(os.path.join(data_path, 'keypoints/H2S.keypoints.train.0.h5'), 'r') as h5_file:
    clip_count = 0
    # Iterate through videos in shard 0
    for video_id, shard_index in shard_dict.items():
        if shard_index == 0:  # Only process videos from shard 0
            video_group = h5_file[video_id]  # Access the group
            clip_order = annotations[video_id]['clip_order']  # Get the clip order
            
            # Initialize new annotation structure for this video
            new_annotations[video_id] = {'clip_order': []}
            new_metadata[video_id] = shard_index  # Copy the shard index for metadata
            
            # Iterate through clips in the video
            for clip_name in clip_order:
                if clip_name in video_group:
                    # Access the clip dataset
                    clip_data = video_group[clip_name][:]
                    
                    # Access the clip metadata from the annotations
                    clip_metadata = annotations[video_id][clip_name]
                    
                    break
            for i in range(max_len):
                # Add the clip data and metadata to the selected_clips list
                clip_name_fake = clip_name + str(i)
                selected_clips.append({
                    'video_id': video_id,
                    'clip_name': clip_name_fake,
                    'data': clip_data,
                    'metadata': clip_metadata
                })
            
                # Update the new annotation for this clip
                new_annotations[video_id]['clip_order'].append(clip_name_fake)
                new_annotations[video_id][clip_name_fake] = clip_metadata
            break
            clip_count += 1
            # Stop when we have 2048 clips
            if clip_count >= max_len:
                break
        # Stop the outer loop when we have 2048 clips
        if clip_count >= max_len:
            break

# Now selected_clips contains the subset of 2048 clips

# Optionally: Save the selected clips into a new HDF5 file
with h5py.File('subset_same_H2S.keypoints.train.0.h5', 'w') as subset_h5:
    for clip in selected_clips:
        # Create a group for each video if it doesn't exist
        if clip['video_id'] not in subset_h5:
            video_group = subset_h5.create_group(clip['video_id'])
        else:
            video_group = subset_h5[clip['video_id']]
        
        # Save the clip data into the new HDF5 file
        video_group.create_dataset(clip['clip_name'], data=clip['data'], dtype='float16')
        
        # Save clip metadata as attributes (you can also save in another format if needed)
        for key, value in clip['metadata'].items():
            video_group[clip['clip_name']].attrs[key] = json.dumps(value)  # Store metadata as JSON
#            print('Saving h5')

# Save the new annotation JSON
with open('subset_same_H2S.annotations.train.json', 'w') as f:
    json.dump(new_annotations, f, indent=4)
    print('Saving annotaitons')

# Save the new metadata JSON (shard indices)
with open('subset_same_H2S.keypoints.train.json', 'w') as f:
    json.dump(new_metadata, f, indent=4)
    print('Saving metadata')

# Subset of 2048 clips, along with new annotations and metadata, is now saved.
