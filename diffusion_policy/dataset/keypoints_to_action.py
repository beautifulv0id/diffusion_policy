from torch.utils.data import Dataset
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import pickle
import os

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..','data'))


def get_split(data, split, seed):
    assert split in ['train', 'valid', 'test', 'all'], f'split {split} not supported.'
    if split != 'all':
        rng = np.random.default_rng(seed)
        indices = np.arange(len(data))
        rng.shuffle(indices)

        n = len(data)
        if split == 'train':
            data = [data[i] for i in indices[:int(n * 0.8)]]
        elif split == 'valid':
            data = [data[i] for i in indices[int(n * 0.8):int(n * 0.9)]]
        elif split == 'test':
            data = [data[i] for i in indices[int(n * 0.9):]]
    return data



def dict_apply(x, func):
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def load_dataset(path):
    dataset = []
    for file in os.listdir(path):
        if file.endswith(".pkl"):
            with open(os.path.join(path, file), "rb") as f:
                data = pickle.load(f)
                data = dict_apply(data, torch.tensor)
                dataset.append(data)
    return dataset

class Keypoints2ActionDataset(Dataset):

    def __init__(self, data_path='rlbench_simple/data', split='train', seed=123456):
        load_path = os.path.join(BASE_PATH, data_path)

        data = load_dataset(load_path)
        self.data = get_split(data, split, seed)

        self.n_context = 29

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx].copy()
        pose = data['obs']['agent_pose']
        pose_pcd = self.agent_pose_to_pcd(pose)

        data['context'] = torch.cat((data['obs']['keypoint_pcd'][0,...],pose_pcd),dim=0)
        data['action'] = data['action'][0,...]
        return data

    def agent_pose_to_pcd(self, pose):
        point_zero = pose[:, :3, -1]
        points_2 = pose[:,:3,:3]*.1 + point_zero[:,None,:]
        return torch.cat((point_zero, points_2.reshape(-1,3)), dim=0)

    def visualize_sample(self, data):
        from geo_rel_policies.utils.plotting import plot_se3, plot_3d

        pose = data['obs']['agent_pose']

        keypoints = data['obs']['keypoint_pcd']

        next_pose = data['action']

        for k in range(pose.shape[0]):
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d", proj_type="ortho")

            plot_se3(pose[k,...], ax)

            plot_se3(next_pose[k,...], ax, color='actor')

            plot_3d(keypoints[k,0,...], ax)


            plt.show()

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset = Keypoints2ActionDataset()

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=0)

    for _, data in enumerate(dataloader):

        dataset.visualize_sample(data)
        plt.show()
