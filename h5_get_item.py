import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import time


class SignLLMDataset(Dataset):
    def __init__(self, data_h5, transform=None):
        self.data_file = h5py.File(data_h5, 'r')
        self.transform = transform

    def __len__(self):
        return len(self.data_file['video'])

    def __getitem__(self, idx):
        video_name = self.data_file['video'][idx].decode('utf-8')
        clip_name = self.data_file['clip'][idx].decode('utf-8')
        features = self.data_file['features'][idx]
        features = torch.tensor(features).float()
        # TODO: Add padding - different length of features
        if self.transform:      # potential augmentations?
            features = self.transform(features)
        return video_name, clip_name, features


if __name__ == '__main__':
    data_h5 = './h5py/my_h5_file.h5'
    train_dataset = SignLLMDataset(data_h5)
    trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
    for video_name, clip_name, features in trainloader:
        print(video_name, clip_name, features)
