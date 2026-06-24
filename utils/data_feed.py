import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import random

scenario_num  = 8
data_root = f'/projappl/project_2013517/DeepSense/datasets/Scenario{scenario_num}/'


def create_samples(root, portion=1., shuffle=False, nat_sort=False):
    f = pd.read_csv(root, na_values='')
    f = f.fillna(-99)
    seq_cols  = sorted([c for c in f.columns if c.startswith('seq') and c[3:].isdigit()],
                       key=lambda x: int(x[3:]))
    beam_cols = sorted([c for c in f.columns if c.startswith('Beam') and c[4:].isdigit()],
                       key=lambda x: int(x[4:]))

    data_samples = []
    pred_beam = []
    inp_beam = []
    for idx, row in f.iterrows():
        lidar_data = row[seq_cols].tolist()

        data_samples.append(lidar_data)
        future_beam_raw = row['Future_Beam1':'Future_Beam3'].tolist()
        future_beam = np.asarray([
            np.argmax(np.loadtxt(f'{data_root}' + str(pwr)[1:])) if str(pwr) != '-99' else -100
            for pwr in future_beam_raw
        ])  # -100 = missing/padded future beam (ignore index)
        #print(future_beam)
        #future_beam = np.asarray([np.argmax(np.loadtxt('/projappl/project_2013517/DeepSense/datasets/scenario9_dev/' + pwr[1:])) for pwr in future_beam]) # scenario 9

        pred_beam.append(future_beam)
        input_beam = row[beam_cols].tolist()

        input_beam = np.asarray([np.argmax(np.loadtxt(f'{data_root}' + pwr[1:])) for pwr in input_beam])
        #input_beam = np.asarray([np.argmax(np.loadtxt('/projappl/project_2013517/DeepSense/datasets/scenario9_dev/' + pwr[1:])) for pwr in input_beam]) # # scenario 9

        inp_beam.append(input_beam)
        
    #print('list is ready')
    num_data = len(data_samples)
    num_data = int(num_data * portion)
    return data_samples[:num_data], inp_beam[:num_data], pred_beam[:num_data]


class DataFeed(Dataset):
    def __init__(self, root_dir, n, init_shuffle=True, portion=1.):

        self.root = root_dir
        self.samples, self.inp_val, self.pred_val = create_samples(
            self.root, shuffle=init_shuffle, portion=portion)
        self.seq_len = n

        # Pre-cache all LiDAR .mat files to avoid repeated disk I/O in __getitem__
        print(f"Pre-caching LiDAR files for {root_dir} ...")
        all_paths = set()
        for sample in self.samples:
            for data in sample[:self.seq_len]:
                all_paths.add(f'{data_root}{data[1:]}')
        self._cache = {path: loadmat(path)['data'][:, 0] for path in all_paths}
        print(f"  Cached {len(self._cache)} unique LiDAR files.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        beam_val = self.pred_val[idx]
        input_beam = self.inp_val[idx]

        sample = sample[:self.seq_len]
        input_beam = input_beam[:self.seq_len]
        out_beam = torch.full((3,), -100, dtype=torch.long)  # default = ignore index
        lidar_val = torch.zeros((self.seq_len, 216))
        input_data = torch.zeros((self.seq_len,))
        for i, data in enumerate(sample):
            lidar_val[i] = torch.from_numpy(self._cache[f'{data_root}{data[1:]}'] / 10 )

        for i, s in enumerate(input_beam):
            input_data[i] = torch.tensor(s, requires_grad=False) - 1

        for i, s in enumerate(beam_val):
            if s != -100:
                out_beam[i] = torch.tensor(s, requires_grad=False) - 1

        return lidar_val, input_data.long(), torch.squeeze(out_beam.long())

if __name__ == "__main__":
    num_classes = 64
    batch_size = 64
    val_batch_size = 64
    train_dir = './utils/train_seqs.csv'
    val_dir = './utils/test_seqs.csv'
    train_loader = DataLoader(DataFeed(train_dir, seq_len, portion=1.), batch_size=batch_size, shuffle=True)
    data = next(iter(train_loader))
    print('done')