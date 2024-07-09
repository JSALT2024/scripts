import h5py

start_index = 0
num_of_items = 5

with h5py.File('./h5py/my_h5_file.h5','r') as fr:
    videos = fr['video'][start_index:start_index+num_of_items]
    clips = fr['clip'][start_index:start_index+num_of_items]
    features = fr['features'][start_index:start_index+num_of_items]

for i in range(0, num_of_items):
    video_name = videos[i].decode('utf-8')
    clip_name = clips[i].decode('utf-8')
    feature = features[i]
    print(video_name, clip_name, feature)
