import os
import h5py
import numpy as np


def save_to_h5(fetures_list_h5, label, index_dataset, chunk_batch, chunk_size):
    if index_dataset == chunk_batch * chunk_size:
        chunk_batch += 1
        fetures_list_h5.resize(chunk_batch * chunk_size, axis=0)
    fetures_list_h5[index_dataset:index_dataset + chunk_size] = label
    index_dataset += chunk_size
    return index_dataset, chunk_batch


if __name__ == '__main__':
    video_ids = ['video1', 'video2']  # list of videos
    clip_ids = [['clip1', 'clip2'], ['clip3']]  # list of lists of clips (one list for each video)
    list_of_features = [[[np.array([1, 2, 3]), np.array([4, 5, 6])], [np.array([7, 8]), np.array([9]), np.array([10])]],
                        [[np.array([11, 12, 13])]]  # list of lists of numpy arrays (one list for each clip)
                        ]
    output_path = "./h5py"
    output_file = 'my_h5_file.h5'

    # h5py file initialization4,
    f_out = h5py.File(os.path.join(output_path, output_file), 'w')

    # special data type for numpy array with variable length
    dt = h5py.vlen_dtype(np.dtype('float64'))

    for i in range(0, len(video_ids)):      # iterating over videos
        video = video_ids[i]
        video_h5 = f_out.create_group(video)
        for idx, clip in enumerate(clip_ids[i]):    # iterating over clips of the video
            # set starting index and starting chunk
            index_dataset = 0
            chunk_batch = 1

            # number of samples in one chunk, set it to the same size as batch size during prediction
            chunk_size = 2

            # ADD PREDICTION OF YOUR MODEL HERE
            # features = model.predict(clip)
            # AND CONCATE THEM TO features
            # OR DO CYCLE OVER THE FRAMES AND SAVE features TO H5 BATCH BY BATCH
            features = list_of_features[i][idx]
            fetures_list_h5 = video_h5.create_dataset(clip, shape=(len(features),), maxshape=(None,), dtype=dt)
            num_full_chunks = len(features) // chunk_size
            last_chunk_size = len(features) % chunk_size
            for c in range(num_full_chunks):
                feature = features[index_dataset:index_dataset + chunk_size]
                index_dataset, chunk_batch = save_to_h5(fetures_list_h5, feature, index_dataset, chunk_batch,
                                                        chunk_size)
            if last_chunk_size > 0:
                feature = features[index_dataset:index_dataset + last_chunk_size]
                index_dataset, chunk_batch = save_to_h5(fetures_list_h5, feature, index_dataset, chunk_batch,
                                                        last_chunk_size)

    f_out.close()
