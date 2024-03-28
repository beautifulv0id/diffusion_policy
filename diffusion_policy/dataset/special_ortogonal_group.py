"""Copyright (c) Dreamfold."""
"""Adapted by Julen Urain
mail: julen@robot-learning.de"""

from torch.utils.data import Dataset
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import os



def get_split(data, split, seed):
    assert split in ['train', 'valid', 'test', 'all'], f'split {split} not supported.'
    if split != 'all':
        rng = np.random.default_rng(seed)
        indices = np.arange(len(data))
        rng.shuffle(indices)

        n = len(data)
        if split == 'train':
            data = data[indices[:int(n * 0.8)]]
        elif split == 'valid':
            data = data[indices[int(n * 0.8):int(n * 0.9)]]
        elif split == 'test':
            data = data[indices[int(n * 0.9):]]
    return data


class SpecialOrthogonalGroup(Dataset):

    def __init__(self, root='data', split='train', seed=12345):
        self.root = os.path.join(os.path.dirname(__file__), 'data')
        data = np.load(f'{self.root}/orthogonal_group.npy').astype('float32')
        self.data = get_split(data, split, seed)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ConditionedSpecialOrthogonalGroup(Dataset):

    def __init__(self, root='data', split='train', seed=12345, random_rotations=True, single_point=False):
        self.root = os.path.join(os.path.dirname(__file__), 'data')
        data = np.load(f'{self.root}/orthogonal_group.npy').astype('float32')
        self.data = get_split(data, split, seed)

        ## Get 3D Points ##
        self.points = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],
                                [.5,.5,0.],[0.,.5,.5],[.5,0.,.5],
                                [.2,.6,.2],[0.,.2,.1],[-.5,-.5,.5],])

        self.context_shape = self.points.shape
        self.data_shape = self.data[0,...].shape

        ## Data Modifiers ##
        self.random_rotations = random_rotations
        self.single_point = single_point

    def __len__(self):
        return len(self.data)

    def get_random_rotation(self):
        return Rotation.random().as_matrix()

    def __getitem__(self, idx):
        if self.single_point:
            idx = 0

        data = self.data[idx]

        R = self.get_random_rotation()
        if self.random_rotations:
            ## Apply Rotation ##
            rot_data = np.einsum('mn,nj->mj',R,data)
            rot_points = np.einsum('mn,bn->bm',R,self.points)
        else:
            rot_data = data
            rot_points = self.points

        d ={
            'data': rot_data,
            'points': rot_points,
            'rotation_matrix': R,
            'prerot_data': data,
            'prerot_points': self.points,
        }

        return d


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from geo_rel_policies.utils.plotting import plot_so3

    dataset = ConditionedSpecialOrthogonalGroup()

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=0)

    for _, data in enumerate(dataloader):

        plot_so3(data['prerot_data'])
        plt.show()
        print(data)
