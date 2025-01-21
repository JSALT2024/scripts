import h5py
import json
import pdb


#pdb.set_trace()

# Paths to the HDF5 shards (modify this list with the actual paths to your files)
'''
h5_shard_paths = ["/scratch/project_465000977/data/How2Sign/features/mae/16-07_21-52-12/H2S.mae.train.0.h5", \
"/scratch/project_465000977/data/How2Sign/features/mae/16-07_21-52-12/H2S.mae.train.1.h5", \
"/scratch/project_465000977/data/How2Sign/features/mae/16-07_21-52-12/H2S.mae.train.2.h5", \
"/scratch/project_465000977/data/How2Sign/features/mae/16-07_21-52-12/H2S.mae.train.3.h5", \
"/scratch/project_465000977/data/How2Sign/features/mae/16-07_21-52-12/H2S.mae.train.4.h5", \
"/scratch/project_465000977/data/How2Sign/features/mae/16-07_21-52-12/H2S.mae.train.5.h5", \
"/scratch/project_465000977/data/How2Sign/features/mae/16-07_21-52-12/H2S.mae.train.6.h5", \
"/scratch/project_465000977/data/How2Sign/features/mae/16-07_21-52-12/H2S.mae.train.7.h5", \
"/scratch/project_465000977/data/How2Sign/features/mae/16-07_21-52-12/H2S.mae.train.8.h5", \
"/scratch/project_465000977/data/How2Sign/features/mae/16-07_21-52-12/H2S.mae.train.9.h5"]
'''
h5_shard_paths = ["/scratch/project_465000977/data/YoutubeASL/features/keypoints/YouTubeASL.keypoints.train.0.h5"]


# Load the metafile with video IDs and their respective shard indices
#with open("/scratch/project_465000977/data/How2Sign/features/mae/16-07_21-52-12/H2S.mae.train.json", "r") as metafile:
with open("/scratch/project_465000977/data/YoutubeASL/features/keypoints/YouTubeASL.keypoints.train.json", "r") as metafile:
    meta_dict = json.load(metafile)

# Dictionary to store the clip lengths
clip_lengths = {}

# Function to extract number of frames from a clip (HDF5 group)
def get_clip_length(h5_file, clip_id):
    clip_group = h5_file[clip_id]
    # Assuming clip data is stored as an HDF5 dataset with the shape (frames, height, width, channels)
    # Replace 'frames' with actual key if needed
    return clip_group.shape[0]  # First dimension is the number of frames

# Iterate over the metafile and open the relevant HDF5 shard to process each video
for video_id, shard_idx in meta_dict.items():
    # Open the corresponding shard
    shard_path = h5_shard_paths[shard_idx]
    with h5py.File(shard_path, 'r') as h5_file:
        # Process each clip for this video
        for clip_id in h5_file[video_id].keys():
            clip_full_id = f"{video_id}/{clip_id}"
            try:
                clip_length = get_clip_length(h5_file, clip_full_id)
                clip_lengths[clip_full_id] = clip_length
            except KeyError:
                print(f"Clip {clip_full_id} not found in {shard_path}")
            except Exception as e:
                print(f"Error processing {clip_full_id}: {str(e)}")

# Save the clip lengths to a JSON file
with open("YTASL.clip_lengths.keypoints.json", "w") as out_file:
    json.dump(clip_lengths, out_file, indent=4)

print("Clip lengths saved to clip_lengths.json")
