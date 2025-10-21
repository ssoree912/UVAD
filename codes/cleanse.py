from glob import glob
import copy
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from torch.nn.utils import prune
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


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
    def __init__(self,
                 dataset_name,
                 uvadmode,
                 *,
                 run_name: str = "default",
                 save_root: str = "artifacts",
                 log_dir: str = "logger",
                 logger: Optional[logging.Logger] = None,
                 device: Optional[torch.device] = None,
                 epochs: int = 10,
                 batch_size: int = 64,
                 optimizer_lr: float = 1e-3,
                 num_workers: int = 4,
                 magnitude_prune: float = 0.0,
                 random_prune: float = 0.0,
                 random_prune_seed: Optional[int] = None,
                 unprune_epoch: Optional[int] = None,
                 eval_config: Optional[str] = None,
                 eval_interval: int = 0,
                 seed: int = 111,
                 config_path: Optional[str] = None):
        super().__init__(dataset_name, uvadmode)

        self.logger = logger or logging.getLogger(f"AppAErecon.{run_name}")
        self.run_name = run_name
        self.save_root = Path(save_root)
        self.log_dir = Path(log_dir)
        self.config_path = Path(config_path) if config_path else None
        self.run_dir = self.save_root / self.dataset_name / self.uvadmode / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.loss_history = []
        self.best_state_dict = None
        self.best_epoch = None
        self.best_loss = None
        self.seed = seed

        self.magnitude_prune = magnitude_prune
        self.random_prune = random_prune
        self.random_prune_seed = random_prune_seed
        self.unprune_epoch = unprune_epoch
        self.eval_config = eval_config
        self.eval_interval = eval_interval
        self.num_epochs = epochs
        self.optimizer_name = 'Adam'
        self.optimizer_lr = optimizer_lr

        fpaths = self.get_app_fpaths()
        if not fpaths:
            raise RuntimeError(f'No patches found for dataset="{self.dataset_name}" '
                               f'and uvadmode="{self.uvadmode}". '
                               'Check that patches are placed correctly.')
        self._log('Run directory: %s', self.run_dir)
        self._log('AppAE training set size: %d patches (%s, %s).',
                  len(fpaths), self.dataset_name, self.uvadmode)
        self._log('Loading patch tensors into batches...')
        self._log('Hyperparameters -> epochs=%d, batch_size=%d, lr=%.6f, workers=%d',
                  self.num_epochs, self.batch_size, self.optimizer_lr, self.num_workers)
        self._log('Pruning -> magnitude=%.2f%%, random=%.2f%%, unprune_epoch=%s (random_seed=%s)',
                  self.magnitude_prune * 100,
                  self.random_prune * 100,
                  str(self.unprune_epoch) if self.unprune_epoch is not None else 'None',
                  str(self.random_prune_seed) if self.random_prune_seed is not None else 'None')

        dataset_train = PatchDataset(fpaths)
        loader_train = DataLoader(dataset_train, batch_size=self.batch_size,
                                  shuffle=True, num_workers=self.num_workers)

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self._log('Using device: %s', self.device)
        self.net = AppAE().to(self.device)
        self.prunable_modules = [m for m in self.net.modules()
                                 if isinstance(m, (nn.Conv2d, nn.Linear))]
        self._apply_initial_pruning()

        opt = optim.Adam(self.net.parameters(), lr=self.optimizer_lr)

        self._log('Starting AppAE training for %d epochs.', self.num_epochs)
        for i_epoch in range(1, self.num_epochs + 1):
            self._log('Epoch %02d/%02d running...', i_epoch, self.num_epochs)
            running_loss = 0.0
            pbar = tqdm(enumerate(loader_train), total=len(loader_train),
                        desc=f'Training AE. Epoch {i_epoch:02d}', leave=False,
                        dynamic_ncols=True, mininterval=1.0, miniters=100)
            for _, batch in pbar:
                xs = batch.to(self.device)

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
            self._log('Epoch %02d average reconstruction loss: %.6f', i_epoch, avg_loss)
            self.loss_history.append(float(avg_loss))

            if self.best_loss is None or avg_loss < self.best_loss:
                self.best_loss = float(avg_loss)
                self.best_epoch = i_epoch
                self.best_state_dict = self._capture_state_dict()
                self._log('New best model at epoch %02d (loss=%.6f).',
                          self.best_epoch, self.best_loss)

            if self.unprune_epoch is not None and i_epoch == self.unprune_epoch:
                self._log('Unpruning network parameters at epoch %02d.', i_epoch)
                self._remove_pruning()

        if self.best_state_dict is None:
            self.best_state_dict = self._capture_state_dict()
            self.best_epoch = self.num_epochs
            self.best_loss = float(self.loss_history[-1])

        self._save_training_artifacts()
        if self.best_state_dict is not None:
            self.net.load_state_dict(self.best_state_dict)

    def infer(self):
        fpaths = self.get_app_fpaths()
        self._log('AppAE inference on %d patches (%s, %s).',
                  len(fpaths), self.dataset_name, self.uvadmode)
        self._log('Beginning inference pass for reconstruction loss estimation.')
        dataset_test = PatchDataset(fpaths)
        loader_train = DataLoader(dataset_test, batch_size=self.batch_size,
                                  shuffle=False, num_workers=self.num_workers)
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

    def _log(self, msg, *args):
        if self.logger:
            self.logger.info(msg, *args)
        else:
            print(msg % args, flush=True)

    def _apply_initial_pruning(self):
        if not self.prunable_modules:
            return
        if self.magnitude_prune > 0:
            self._log('Applying magnitude pruning: %.2f%%', self.magnitude_prune * 100)
            for module in self.prunable_modules:
                prune.l1_unstructured(module, name='weight', amount=self.magnitude_prune)
        if self.random_prune > 0:
            if self.random_prune_seed is not None:
                torch.manual_seed(self.random_prune_seed)
                np.random.seed(self.random_prune_seed)
            self._log('Applying random pruning: %.2f%% (seed=%s)',
                      self.random_prune * 100,
                      self.random_prune_seed if self.random_prune_seed is not None else 'default')
            for module in self.prunable_modules:
                prune.random_unstructured(module, name='weight', amount=self.random_prune)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

    def _remove_pruning(self):
        removed_any = False
        for module in self.prunable_modules:
            if hasattr(module, 'weight_orig'):
                prune.remove(module, 'weight')
                removed_any = True
        if removed_any:
            self._log('Removed pruning reparameterizations; model is now dense.')

    def _capture_state_dict(self):
        model_copy = copy.deepcopy(self.net).to('cpu')
        for module in model_copy.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                try:
                    prune.remove(module, 'weight')
                except ValueError:
                    continue
        return {k: v.detach().cpu() for k, v in model_copy.state_dict().items()}

    def _compose_run_command(self) -> str:
        parts = [
            "python main1_pseudoanomaly.py",
            f"--dataset_name={self.dataset_name}",
            f"--uvadmode={self.uvadmode}",
            "--mode=app",
            f"--seed={self.seed}",
            f"--run_name={self.run_name}",
            f"--epochs={self.num_epochs}",
            f"--batch_size={self.batch_size}",
            f"--lr={self.optimizer_lr}",
            f"--num_workers={self.num_workers}",
            f"--save_root={self.save_root}",
            f"--log_dir={self.log_dir}",
            f"--magnitude_prune={self.magnitude_prune}",
            f"--random_prune={self.random_prune}",
            f"--device={self.device}",
        ]
        if self.random_prune_seed is not None:
            parts.append(f"--random_prune_seed={self.random_prune_seed}")
        if self.unprune_epoch is not None:
            parts.append(f"--unprune_epoch={self.unprune_epoch}")
        if self.config_path is not None:
            parts.append(f"--config={self.config_path}")
        if self.eval_config is not None:
            parts.append(f"--eval_config={self.eval_config}")
        if self.eval_interval is not None:
            parts.append(f"--eval_interval={self.eval_interval}")
        return ' '.join(str(p) for p in parts)

    def _save_training_artifacts(self):
        if not self.loss_history:
            return

        final_state = self._capture_state_dict()
        final_path = self.run_dir / 'appaerecon_final.pkl'
        torch.save(final_state, final_path)

        best_path = None
        if self.best_state_dict is not None:
            best_path = self.run_dir / 'appaerecon_best.pkl'
            torch.save(self.best_state_dict, best_path)

        # Compatibility copies
        shutil.copyfile(final_path, self.run_dir / 'final.pkl')
        if best_path is not None:
            shutil.copyfile(best_path, self.run_dir / 'best_auc.pkl')

        log_file = None
        if self.logger:
            for handler in self.logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    log_file = handler.baseFilename
                    break

        metadata = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'dataset_name': self.dataset_name,
            'uvadmode': self.uvadmode,
            'run_name': self.run_name,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'optimizer': self.optimizer_name,
            'optimizer_lr': float(self.optimizer_lr),
            'seed': self.seed,
            'device': str(self.device),
            'best_epoch': self.best_epoch,
            'best_loss': float(self.best_loss) if self.best_loss is not None else None,
            'final_loss': float(self.loss_history[-1]),
            'loss_history': self.loss_history,
            'num_parameters': int(sum(p.numel() for p in self.net.parameters())),
            'pruning': {
                'magnitude': self.magnitude_prune,
                'random': self.random_prune,
                'random_seed': self.random_prune_seed,
                'unprune_epoch': self.unprune_epoch,
            },
            'evaluation': {
                'config': str(self.eval_config) if self.eval_config else None,
                'interval': self.eval_interval,
                'notes': 'Per-epoch AUROC evaluation not implemented due to compute cost.'
            },
            'checkpoints': {
                'best': str(best_path) if best_path else None,
                'final': str(final_path),
                'best_auc': str(self.run_dir / 'best_auc.pkl'),
                'final_alias': str(self.run_dir / 'final.pkl')
            },
            'paths': {
                'save_root': str(self.save_root),
                'run_dir': str(self.run_dir),
                'log_dir': str(self.log_dir),
                'config_path': str(self.config_path) if self.config_path else None,
            },
            'log_file': log_file,
            'run_command': self._compose_run_command()
        }

        config_path = self.run_dir / 'appaerecon_training_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        self._log('Saved final checkpoint to %s', final_path)
        if best_path is not None:
            self._log('Saved best checkpoint (epoch %02d) to %s', self.best_epoch, best_path)
        self._log('Training metadata saved to %s', config_path)


class fGMM(Cleanse):
    def __init__(self, dataset_name, uvadmode, tr_f, N=12, logger: Optional[logging.Logger] = None):
        super().__init__(dataset_name, uvadmode)
        self.logger = logger or logging.getLogger(f'fGMM.{dataset_name}.{uvadmode}')

        self._log('Fitting GMM with %d samples (%s, %s).',
                  tr_f.shape[0], self.dataset_name, self.uvadmode)
        self._log('Starting GaussianMixture training with %d components.', N)
        self.gmm = GaussianMixture(n_components=N, max_iter=300).fit(tr_f)

    def infer(self, tr_f):
        self._log('Inference on %d samples with trained GMM.', tr_f.shape[0])
        return -self.gmm.score_samples(tr_f)

    def _log(self, msg, *args):
        if self.logger:
            self.logger.info(msg, *args)
        else:
            print(msg % args, flush=True)
