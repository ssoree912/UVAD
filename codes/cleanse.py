from glob import glob

import numpy as np
import torch
import torch.optim as optim
from sklearn.mixture import GaussianMixture
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import torch.nn as nn


class AppAE(nn.Module):
    def __init__(self):
        super(AppAE, self).__init__()
        D1 = 32
        D2 = 64
        D3 = 128
        D4 = 128

        self.conv1 = nn.Conv2d(1, D1, kernel_size=(3, 3), padding=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(D1, D2, kernel_size=(3, 3), padding=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3a = nn.Conv2d(D2, D3, kernel_size=(3, 3), padding=(1, 1))
        self.conv3b = nn.Conv2d(D3, D4, kernel_size=(3, 3), padding=(1, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv4a = nn.Conv2d(D4, D4, kernel_size=(3, 3), padding=(1, 1))
        self.conv4b = nn.Conv2d(D4, D4, kernel_size=(3, 3), padding=(1, 1))  # (512 -> 64)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        #######

        self.t_conv1a = nn.ConvTranspose2d(D4, D4, kernel_size=(3, 3), stride=1, padding=1)
        self.t_conv1b = nn.ConvTranspose2d(D4, D3, kernel_size=(2, 2), stride=(2, 2))

        self.t_conv2a = nn.ConvTranspose2d(D3, D2, kernel_size=(3, 3), stride=1, padding=1)
        self.t_conv2b = nn.ConvTranspose2d(D2, D1, kernel_size=(2, 2), stride=(2, 2))
        self.t_conv3 = nn.ConvTranspose2d(D1, 1, kernel_size=(2, 2), stride=(2, 2))

        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 1, 1))
        self.relu = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))

        #########

        h = self.relu(self.t_conv1a(h))
        h = self.relu(self.t_conv1b(h))

        h = self.relu(self.t_conv2a(h))
        h = self.relu(self.t_conv2b(h))
        h = self.sigmoid(self.t_conv3(h))

        return h


class Cleanse:
    def __init__(self, dataset_name, uvadmode):
        self.dataset_name = dataset_name
        self.uvadmode = uvadmode

    def get_app_fpaths(self):
        dataset_name = self.dataset_name
        mode = self.uvadmode
        pattern1 = f'patches/{dataset_name}/train/*/*/*.npy'
        pattern2 = f'patches/{dataset_name}/test/*/*/*.npy'
        fpaths1 = sorted(glob(pattern1))
        fpaths2 = sorted(glob(pattern2))

        if mode == 'merge':
            fpaths = fpaths1 + fpaths2
        elif mode == 'partial':
            fpaths = fpaths2
        else:
            raise ValueError()

        return fpaths


class PatchDataset(Dataset):
    def __init__(self, l_paths):
        super().__init__()
        self.l_paths = l_paths

    def __len__(self):
        return len(self.l_paths)

    def __getitem__(self, index):
        fpath = self.l_paths[index]
        patch = np.load(fpath)
        patch = np.expand_dims(patch, axis=-1)

        patch = patch.astype(np.float32)
        patch /= 255
        patch = np.transpose(patch, [2, 0, 1])
        return patch


class AppAErecon(Cleanse):
    def __init__(self, dataset_name, uvadmode):
        super().__init__(dataset_name, uvadmode)

        fpaths = self.get_app_fpaths()
        if not fpaths:
            raise RuntimeError(f'No patches found for dataset="{self.dataset_name}" '
                               f'and uvadmode="{self.uvadmode}". '
                               'Check that patches are placed correctly.')
        print(f'[INFO] AppAE training set size: {len(fpaths)} patches '
              f'({self.dataset_name}, {self.uvadmode}).')
        print('[INFO] Loading patch tensors into batches...')
        dataset_train = PatchDataset(fpaths)
        loader_train = DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=4)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[INFO] Using device: {self.device}')
        self.net = AppAE().to(self.device)
        opt = optim.Adam(self.net.parameters())

        self.num_epochs = 10
        print(f'[INFO] Starting AppAE training for {self.num_epochs} epochs.')
        for i_epoch in range(1, self.num_epochs + 1):
            print(f'[INFO] Epoch {i_epoch:02d}/{self.num_epochs} running...')
            running_loss = 0.0
            pbar = tqdm(enumerate(loader_train), total=len(loader_train),
                        desc=f'Training AE. Epoch {i_epoch:02d}', leave=True, dynamic_ncols=True)
            for i_batch, batch in pbar:
                xs = batch
                xs = xs.to(self.device)

                opt.zero_grad()

                xhs = self.net(xs)
                loss = (xhs - xs) ** 2
                loss = loss.mean()
                loss.backward()

                opt.step()
                loss_value = loss.item()
                running_loss += loss_value * xs.size(0)
                pbar.set_postfix(loss=f'{loss_value:.5f}')
            avg_loss = running_loss / len(dataset_train)
            print(f'[INFO] Epoch {i_epoch:02d} average reconstruction loss: {avg_loss:.6f}')

    def infer(self):
        fpaths = self.get_app_fpaths()
        print(f'[INFO] AppAE inference on {len(fpaths)} patches '
              f'({self.dataset_name}, {self.uvadmode}).')
        print('[INFO] Beginning inference pass for reconstruction loss estimation.')
        dataset_test = PatchDataset(fpaths)
        loader_train = DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=4)
        ret = []
        for i_batch, batch in tqdm(enumerate(loader_train), total=len(loader_train),
                                   desc='Inferring AE scores', leave=True, dynamic_ncols=True):
            xs = batch
            xs = xs.to(self.device)

            xhs = self.net(xs)
            loss = (xhs - xs) ** 2
            loss = loss.mean(dim=(1, 2, 3))
            ret.append(loss.detach().cpu().numpy())

        ret = np.concatenate(ret)
        return ret


class fGMM(Cleanse):
    def __init__(self, dataset_name, uvadmode, tr_f, N=12):
        super().__init__(dataset_name, uvadmode)

        print(f'[INFO] Fitting GMM with {tr_f.shape[0]} samples '
              f'({self.dataset_name}, {self.uvadmode}).')
        print(f'[INFO] Starting GaussianMixture training with {N} components.')
        self.gmm = GaussianMixture(n_components=N, max_iter=300).fit(tr_f)

    def infer(self, tr_f):
        print(f'[INFO] Inference on {tr_f.shape[0]} samples with trained GMM.')
        return -self.gmm.score_samples(tr_f)
