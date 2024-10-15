import argparse
import random

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score

from codes import bbox, signal, vad
from codes.utils import load_config, task

random.seed(111)
np.random.seed(111)
torch.manual_seed(111)
torch.use_deterministic_algorithms(True)


def gaussian_video_d(d, sigma=3):
    ret = {}
    for k in d:
        ret[k] = {}
        for vname in d[k]:
            ret[k][vname] = gaussian_filter1d(d[k][vname], sigma)
    return ret


def calc_AUROC_d(dataset_name, d):
    lengths = np.load(f'meta/test_lengths_{dataset_name}.npy')
    labels = np.load(f'meta/frame_labels_{dataset_name}.npy')
    vnames = vad.get_vnames(dataset_name, mode='test')
    lengths = np.concatenate([[0], lengths])

    ret = {}

    for k in d:
        ret[k] = {}
        for i_s, i_e, vname in zip(lengths[:-1], lengths[1:], vnames):
            arr = d[k][vname]
            y_true = np.concatenate([[0], labels[i_s: i_e], [1]])
            y_pred = np.concatenate([[0], arr, [1000000]])
            AUROC = roc_auc_score(y_true, y_pred)
            ret[k][vname] = AUROC * 100
    return ret


def main(args, config):
    dataset_name = args.dataset_name
    uvadmode = args.mode

    cf_sig = config.signals

    with task('Load features'):
        d = {'te_scorebbox': {}, }

        d['te_scorebbox']['mot'] = signal.MotSignal(dataset_name, cf_sig.mot, uvadmode).get()
        d['te_scorebbox']['app'] = signal.AppSignal(dataset_name, cf_sig.app, uvadmode).get()

    with task():
        obj = bbox.VideosFrameBBs.load(dataset_name, mode='test')

        keys_use = []
        for k in config.signals:
            try:
                if config.signals[k].use:
                    keys_use.append(k)
            except KeyError:
                continue

        for key in keys_use:
            obj.add_signal(key, d['te_scorebbox'][key])

    d_scores_save = {}
    d_scores = obj.get_framesignal_maximum()
    d_scores_save.update(d_scores)
    if config.postprocess.sigma > 0:
        d_scores = gaussian_video_d(d_scores, config.postprocess.sigma)

    d_scores_save['all'] = d_scores['all']
    d['te_score'] = d_scores_save

    d_AUROC = calc_AUROC_d(dataset_name, d_scores)
    d['AUROC'] = d_AUROC['all']

    AUROC = np.mean(list(d_AUROC['all'].values()))
    print(f'AUROC {args.dataset_name} ({args.mode}): {AUROC:.1f}', end='')
    if args.quiet:
        print()
    else:
        print(' ', args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    parser.add_argument("--dataset_name", default='ped2', choices=['shanghaitech', 'avenue', 'ped2'])
    parser.add_argument("--mode", default='partial', choices=['partial', 'merge'])

    parser.add_argument("--quiet", action='store_true', default=True)
    parser.add_argument("--override", default='{}', type=str)
    args_ = parser.parse_args()
    config_ = load_config(args_.config, args_)
    if not args_.quiet:
        print(args_)

    main(args_, config_)
