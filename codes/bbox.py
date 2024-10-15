import numpy as np

from . import utils, vad
from .utils import save_json, filter_keys

__all__ = ['BB', 'FrameBBs', 'VideoFrameBBs']


def normalize(arr, minv=None, maxv=None):
    if minv is None:
        minv = np.min(arr)
    if maxv is None:
        maxv = np.max(arr)
    arr = np.asarray(arr)
    if minv == maxv:
        return arr - minv

    return (arr - minv) / (maxv - minv)


########
class BB:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.signals = {}
        self.info = {}

    def to_dict(self, keys=None):
        d = {
            'x1': float(self.x1),
            'x2': float(self.x2),
            'y1': float(self.y1),
            'y2': float(self.y2),
        }
        d.update(self.signals)
        d.update(self.info)

        if keys is not None:
            d = filter_keys(d, keys)
        return d

    def add_signal(self, key, v):
        self.signals[key] = v

    def get_signal(self, weights=None):
        d = self.signals.copy()

        vs = list(self.signals.values())
        if len(vs) == 0:
            raise KeyError('No signal in bbox')

        combined = np.mean(vs)
        d.update({
            'all': combined
        })
        return d

    def get_vector(self, key):
        return [self.to_dict()[key]]

    @property
    def xxyy(self):
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)


############


class ABCPropagate:
    def __init__(self, d, dataset_name):
        self.d = d
        self.dataset_name = dataset_name

    def to_dict(self, keys=None):
        return {k: v.to_dict(keys=keys) for k, v in self.d.items()}

    def add_signal(self, key, d):
        for k in d:
            try:
                self.d[k].add_signal(key, d[k])
            except KeyError:
                print(self.__class__.__name__)
                print(f'Key: {k}')
                print(self.d.keys())
                print(d.keys())
                raise

    def get_signal(self, weights=None):
        return {k: v.get_signal(weights) for k, v in self.d.items()}

    def get_vector(self, key):
        vecs = [self.d[k].get_vector(key) for k in sorted(self.d.keys())]
        vecs = np.concatenate(vecs)
        return vecs


class FrameBBs(ABCPropagate):
    def __init__(self, d, dataset_name, vname, t):
        super().__init__(d, dataset_name)
        self.vname = vname
        self.t = t

    @staticmethod
    def from_mat(mat, dataset_name, vname, t):
        d = {f'{i:05d}': BB(*vec) for i, vec in enumerate(mat)}
        return FrameBBs(d, dataset_name, vname, t)

    ######################


class VideoFrameBBs(ABCPropagate):
    def __init__(self, l_meta, dataset_name, vname):
        self.T = len(l_meta)
        self.l_meta = l_meta
        self.vname = vname
        d = {f'{t:05d}': FrameBBs.from_mat(mat, dataset_name, vname, t) for t, mat in enumerate(l_meta)}
        super().__init__(d, dataset_name)

    def get_framesignal_maximum(self, key):
        ret = []
        for t in range(self.T):
            frame = self.d[f'{t:05d}']
            frame.get_signal()

            bb_scores = []
            for bb in frame.d.values():
                d_signal = bb.get_signal()
                try:
                    bb_scores.append(d_signal[key])
                except KeyError:
                    pass
            if len(bb_scores):
                score_max = max(bb_scores)
            else:
                score_max = 0
            ret.append(score_max)

        ret = normalize(ret)
        return ret


######################

class VideosFrameBBs(ABCPropagate):
    def __init__(self, videos, dataset_name, mode):
        super().__init__(videos, dataset_name)
        fpath_length = f'meta/{mode}_lengths_{dataset_name}.npy'
        self.lengths = np.load(fpath_length)
        self.keys_signal = set()

    def add_signal(self, key, d):
        self.keys_signal.add(key)
        res = utils.split_by_videos_bboxs(self.dataset_name, d)

        for vname in res:
            self.d[vname].add_signal(key, res[vname])

    ###

    def get_framesignal_maximum(self, keys=None):
        if keys is None:
            keys = list(self.keys_signal) + ['all']
        return {
            key: {
                k: v.get_framesignal_maximum(key=key)
                for k, v in self.d.items()
            }
            for key in keys
        }

    ###

    @staticmethod
    def load(dataset_name, mode='test'):
        vnames = vad.get_vnames(dataset_name, mode)

        fpath = f'meta/{mode}_bboxes_{dataset_name}.npy'
        fpath_length = f'meta/{mode}_lengths_{dataset_name}.npy'

        l = np.load(fpath, allow_pickle=True)
        l_length = np.load(fpath_length)
        l_length = [0] + list(l_length)

        ret = {}
        for vname, i, e in zip(vnames, l_length[:-1], l_length[1:]):
            ret[vname] = VideoFrameBBs(l[i: e], dataset_name, vname)

        return VideosFrameBBs(ret, dataset_name, mode)

    def save(self, fpath):
        obj = self.to_dict()
        save_json(obj, fpath)
