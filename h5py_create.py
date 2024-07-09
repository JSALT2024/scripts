import os
import h5py
import numpy as np


def save_to_h5(video_h5, clip_h5, fetures_list_h5, video_id, clip_id, label, index_dataset, chunk_batch,
               chunk_size):
    if index_dataset == chunk_batch * chunk_size:
        chunk_batch += 1
        video_h5.resize(chunk_batch * chunk_size, axis=0)
        clip_h5.resize(chunk_batch * chunk_size, axis=0)
        fetures_list_h5.resize(chunk_batch * chunk_size, axis=0)
    video_h5[index_dataset:index_dataset+chunk_size] = video_id
    clip_h5[index_dataset:index_dataset+chunk_size] = clip_id
    fetures_list_h5[index_dataset:index_dataset+chunk_size] = label
    index_dataset += chunk_size
    return index_dataset, chunk_batch


if __name__ == '__main__':
    video_ids = ['video1', 'video1', 'video1', 'video2', 'video2', 'video3']
    clip_ids = ['clip1', 'clip2', 'clip3', 'clip4', 'clip5', 'clip6']
    labels = [np.array([0, 1, 2]), np.array([1, 5]), np.array([1.6, 16, 0.5]), np.array([0.1, 0.2, 0.3]),
              np.array([0.1, 0.2]), np.array([0.1, 0.2, 0.3, 42])]
    output_path = "./h5py"
    output_file = 'my_h5_file.h5'

    # number of samples in one chunk
    chunk_size = 2

    # set starting index and starting chunk
    index_dataset = 0
    chunk_batch = 1

    # h5py file initialization
    f_out = h5py.File(os.path.join(output_path, output_file), 'w')

    # special data type for numpy array with variable length
    dt = h5py.vlen_dtype(np.dtype('float64'))

    video_h5 = f_out.create_dataset('video', shape=(chunk_size,), maxshape=(None,),
                                    dtype=h5py.string_dtype(encoding='utf-8'))
    clip_h5 = f_out.create_dataset('clip', shape=(chunk_size,), maxshape=(None,),
                                   dtype=h5py.string_dtype(encoding='utf-8'))
    fetures_list_h5 = f_out.create_dataset('features', shape=(chunk_size,), maxshape=(None,), dtype=dt)

    while True:
        video_id = video_ids[index_dataset:index_dataset+chunk_size]
        clip_id = clip_ids[index_dataset:index_dataset+chunk_size]
        label = labels[index_dataset:index_dataset+chunk_size]
        index_dataset, chunk_batch = save_to_h5(video_h5, clip_h5, fetures_list_h5, video_id, clip_id, label,
                                                index_dataset, chunk_batch, chunk_size)
        if index_dataset >= len(video_ids):
            break

    f_out.close()
