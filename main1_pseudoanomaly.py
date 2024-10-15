import os
import argparse
import numpy as np

from codes.utils import load_json
from codes import cleanse, featurebank
import torch
from sklearn.metrics import roc_auc_score
import random

random.seed(111)
np.random.seed(111)
torch.manual_seed(111)
torch.use_deterministic_algorithms(True)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='avenue')
parser.add_argument('--uvadmode', default='merge', choices=['merge', 'partial'])
parser.add_argument('--mode', default='app', choices=['app', 'mot'])
parser.add_argument('--gmm_n', default=12, type=int)

args = parser.parse_args()


def main():
    dataset_name = args.dataset_name
    uvadmode = args.uvadmode

    dpath = f'features/{dataset_name}/cleansescores'

    if args.mode == 'app':
        ret = cleanse.AppAErecon(dataset_name, uvadmode).infer()
        fpath = f'{dpath}/{uvadmode}_aerecon_flat.npy'
    elif args.mode == 'mot':
        tr_f = featurebank.get(dataset_name, 'mot', 'train', uvadmode=uvadmode).astype(np.float32)
        ret = cleanse.fGMM(dataset_name, uvadmode, tr_f, args.gmm_n).infer(tr_f)
        fpath = f'{dpath}/{uvadmode}_velo_fgmm_flat.npy'

    else:
        raise ValueError()

    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    np.save(fpath, ret)


if __name__ == '__main__':
    main()
