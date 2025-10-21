from glob import glob
import copy
import json
from pathlib import Path
from datetime import datetime

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

        self.batch_size = 64
        self.checkpoint_dir = Path('artifacts') / self.dataset_name / self.uvadmode
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.loss_history = []
        self.best_state_dict = None
        self.best_epoch = None

        fpaths = self.get_app_fpaths()
        if not fpaths:
            raise RuntimeError(f'No patches found for dataset="{self.dataset_name}" '
                               f'and uvadmode="{self.uvadmode}". '
                               'Check that patches are placed correctly.')
        print(f'[INFO] AppAE training set size: {len(fpaths)} patches '
              f'({self.dataset_name}, {self.uvadmode}).', flush=True)
        print('[INFO] Loading patch tensors into batches...', flush=True)
        dataset_train = PatchDataset(fpaths)
        loader_train = DataLoader(dataset_train, batch_size=self.batch_size,
                                  shuffle=True, num_workers=4)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[INFO] Using device: {self.device}', flush=True)
        self.net = AppAE().to(self.device)
        opt = optim.Adam(self.net.parameters())
        self.optimizer_lr = opt.param_groups[0]['lr']

        self.num_epochs = 10
        print(f'[INFO] Starting AppAE training for {self.num_epochs} epochs.', flush=True)
        best_loss = None
        for i_epoch in range(1, self.num_epochs + 1):
            print(f'[INFO] Epoch {i_epoch:02d}/{self.num_epochs} running...', flush=True)
            running_loss = 0.0
            pbar = tqdm(enumerate(loader_train), total=len(loader_train),
                        desc=f'Training AE. Epoch {i_epoch:02d}', leave=False,
                        dynamic_ncols=True, mininterval=1.0, miniters=100)
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
                pbar.set_postfix_str(f'loss={loss_value:.5f}')
            avg_loss = running_loss / len(dataset_train)
            print(f'[INFO] Epoch {i_epoch:02d} average reconstruction loss: {avg_loss:.6f}',
                  flush=True)
            self.loss_history.append(float(avg_loss))
            if best_loss is None or avg_loss < best_loss:
                best_loss = float(avg_loss)
                self.best_epoch = i_epoch
                self.best_state_dict = {k: v.detach().cpu().clone()
                                        for k, v in self.net.state_dict().items()}

        self._save_training_artifacts(best_loss)
        if self.best_state_dict is not None:
            self.net.load_state_dict(self.best_state_dict)

    def infer(self):
        fpaths = self.get_app_fpaths()
        print(f'[INFO] AppAE inference on {len(fpaths)} patches '
              f'({self.dataset_name}, {self.uvadmode}).', flush=True)
        print('[INFO] Beginning inference pass for reconstruction loss estimation.', flush=True)
        dataset_test = PatchDataset(fpaths)
        loader_train = DataLoader(dataset_test, batch_size=self.batch_size,
                                  shuffle=False, num_workers=4)
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

    def _save_training_artifacts(self, best_loss):
        if not self.loss_history:
            return

        final_path = self.checkpoint_dir / 'appaerecon_final.pkl'
        final_state = self._state_dict_to_cpu(self.net.state_dict())
        torch.save(final_state, final_path)

        best_path = None
        if self.best_state_dict is not None:
            best_path = self.checkpoint_dir / 'appaerecon_best.pkl'
            torch.save(self.best_state_dict, best_path)

        metadata = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'dataset_name': self.dataset_name,
            'uvadmode': self.uvadmode,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'optimizer': 'Adam',
            'optimizer_lr': self.optimizer_lr,
            'run_command': (
                f'python main1_pseudoanomaly.py '
                f'--dataset_name={self.dataset_name} '
                f'--uvadmode={self.uvadmode} --mode=app'
            ),
            'seeds': {
                'python_random': 111,
                'numpy': 111,
                'torch': 111,
                'torch_deterministic': True
            },
            'device': str(self.device),
            'best_epoch': self.best_epoch,
            'best_loss': float(best_loss) if best_loss is not None else None,
            'final_loss': float(self.loss_history[-1]),
            'loss_history': self.loss_history,
            'num_parameters': int(sum(p.numel() for p in self.net.parameters())),
            'checkpoints': {
                'best': str(best_path) if best_path else None,
                'final': str(final_path)
            }
        }

        config_path = self.checkpoint_dir / 'appaerecon_training_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        print(f'[INFO] Saved final checkpoint to {final_path}', flush=True)
        if best_path is not None:
            print(f'[INFO] Saved best checkpoint (epoch {self.best_epoch}) to {best_path}',
                  flush=True)
        print(f'[INFO] Training metadata saved to {config_path}', flush=True)

    @staticmethod
    def _state_dict_to_cpu(state_dict):
        return {k: v.detach().cpu() for k, v in state_dict.items()}


class fGMM(Cleanse):
    def __init__(self, dataset_name, uvadmode, tr_f, N=12):
        super().__init__(dataset_name, uvadmode)

        print(f'[INFO] Fitting GMM with {tr_f.shape[0]} samples '
              f'({self.dataset_name}, {self.uvadmode}).', flush=True)
        print(f'[INFO] Starting GaussianMixture training with {N} components.', flush=True)
        self.gmm = GaussianMixture(n_components=N, max_iter=300).fit(tr_f)

    def infer(self, tr_f):
        print(f'[INFO] Inference on {tr_f.shape[0]} samples with trained GMM.', flush=True)
        return -self.gmm.score_samples(tr_f)
