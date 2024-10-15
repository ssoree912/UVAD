import numpy as np
import time
import faiss
import yaml
import json
from contextlib import contextmanager
import _pickle as p

from . import vad

__all__ = ['FaissKMeans', 'split_by_videos', 'split_by_videos_bboxs', 'attrdict_r', 'apply_default', 'apply_override',
           'load_config', 'load_json', 'save_json', 'filter_keys', 'task', 'save_binary', 'load_binary']


class BetterDict(dict):
    def filt_keys(self, prefix=''):
        return self.__class__({k: v for k, v in self.items() if k.startswith(prefix)})

    def apply(self, func):
        for k, v in self.items():
            self[k] = func(v)
        return self

    def applyarr(self, func):
        for k, v in self.items():
            if isinstance(v, list) or isinstance(v, np.ndarray):
                self[k] = func(v)
        return self

    def as_dict(self):
        return dict(self)


class attrdict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    as_dict = BetterDict.as_dict
    filt_keys = BetterDict.filt_keys
    apply = BetterDict.apply
    applyarr = BetterDict.applyarr


@contextmanager
def task(*args, verbose=False, **kwargs):
    if verbose:
        s = time.time()

    yield

    if verbose:
        e = time.time()
        print(f'Took {e - s:.1s}')


def load_json(fpath, encoding=None):
    with open(fpath, 'r', encoding=encoding) as f:
        return json.load(f)


def save_json(d, fpath, ensure_ascii=True, encoding=None):
    with open(fpath, 'w', encoding=encoding) as f:
        json.dump(d, f, indent=4, ensure_ascii=ensure_ascii)


def load_binary(fpath, encoding='ASCII'):
    with open(fpath, 'rb') as f:
        return p.load(f, encoding=encoding)


def save_binary(d, fpath):
    with open(fpath, 'wb') as f:
        p.dump(d, f)


def filter_keys(d: dict, keys: list) -> dict:
    return {
        k: v for k, v in d.items() if k in keys
    }


class FaissKMeans:
    def __init__(self, n_clusters=10, n_init=100, max_iter=300):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X, y):
        self.kmeans = faiss.Kmeans(d=X.shape[1],
                                   k=self.n_clusters,
                                   niter=self.max_iter,
                                   nredo=self.n_init)
        self.kmeans.train(X.astype(np.float32))
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]

    def predict(self, X):
        return self.kmeans.index.search(X.astype(np.float32), 1)[1]


def split_by_videos(dataset_name, arr):
    lengths = np.load(f'meta/test_lengths_{dataset_name}.npy')
    lengths = np.concatenate([[0], lengths])
    vnames = vad.get_vnames(dataset_name, mode='test')

    ret = {}
    for i_s, i_e, vname in zip(lengths[:-1], lengths[1:], vnames):
        ret[vname] = arr[i_s: i_e]

    return ret


def split_by_videos_bboxs(dataset_name, arr):
    vnames = vad.get_vnames(dataset_name, mode='test')
    ret = split_by_videos(dataset_name, arr)

    for vname in vnames:
        d = {}
        mat = ret[vname]
        T = len(mat)
        for t in range(T):
            d[f'{t:05d}'] = {f'{bi:05d}': v for bi, v in enumerate(mat[t])}
        ret[vname] = d
    return ret


def attrdict_r(d):
    keys = sorted(d.keys())
    for k in keys:
        if isinstance(d[k], dict):
            d[k] = attrdict_r(d[k])
    return attrdict(d)


def apply_default(d):
    keys = sorted(d.keys())
    try:
        default = d['default']
    except KeyError:
        return d

    for k in keys:
        if k == 'default':
            continue
        for kk in default:
            if kk not in d[k].keys():
                d[k][kk] = default[kk]
    return d


def apply_override(child, parent):
    keys = sorted(parent.keys())
    for k in keys:
        if isinstance(parent[k], dict):
            try:
                c = child[k]
            except KeyError:
                c = dict()
            child[k] = apply_override(c, parent[k])
        else:
            child[k] = parent[k]

    return child


def load_config(fpath, args):
    with open(fpath) as f:
        d = yaml.load(f, Loader=yaml.FullLoader)

    d['signals'] = apply_default(d['signals'])

    d2 = eval(args.override)
    d = apply_override(d, d2)

    return attrdict_r(d)
