import numpy as np

from . import featurebank
from . import grader


def calc_minmax(arr):
    minv = np.min(arr)
    maxv = np.max(arr)
    return minv, maxv


def normalize(arr, minv, maxv):
    ret = np.asarray([np.clip((l - minv) / (maxv - minv), 0, 1) for l in arr], dtype=object)
    return ret


class ABCSignal:
    def __init__(self, conf):
        self.conf = conf


class MotSignal(ABCSignal):
    def __init__(self, dataset_name, conf, uvadmode='partial'):
        super().__init__(conf)
        feature_name = 'mot'
        self.tr_f = featurebank.get(dataset_name, feature_name, 'train', uvadmode)

        if conf.cleanse_scorename:
            cleanse_scorepath = f'features/{dataset_name}/cleansescores/{uvadmode}_{conf.cleanse_scorename}_flat.npy'
            cleanse_score = np.load(cleanse_scorepath, allow_pickle=True)
            thres = np.percentile(cleanse_score, conf.percentile_cleanse)
            mask = cleanse_score <= thres
            self.tr_f = self.tr_f[mask]

        if conf.coreset < 100:
            N = int(len(self.tr_f) * conf.coreset / 100)
            inds = np.random.choice(np.arange(len(self.tr_f)), N, replace=False)
            self.tr_f = self.tr_f[inds]

        self.te_f = featurebank.get(dataset_name, feature_name, 'test', uvadmode)
        self.grader = None

    def get(self):
        gr = grader.KNNGrader(self.tr_f, K=self.conf.NN, key='mot')
        tr_scorebbox = gr.grade_flat(self.tr_f)
        minv, maxv = calc_minmax(tr_scorebbox)
        te_scorebbox = gr.grade(self.te_f)
        return normalize(te_scorebbox, minv, maxv)


class AppSignal(ABCSignal):
    def __init__(self, dataset_name, conf, uvadmode='partial', verbose=False):
        super().__init__(conf)

        feature_name = 'app'
        self.dataset_name = dataset_name
        self.uvadmode = uvadmode
        self.tr_f = featurebank.get(dataset_name, feature_name, 'train', uvadmode)

        if conf.cleanse_scorename:
            cleanse_scorepath = f'features/{dataset_name}/cleansescores/{uvadmode}_{conf.cleanse_scorename}_flat.npy'
            cleanse_score = np.load(cleanse_scorepath, allow_pickle=True)

            thres = np.percentile(cleanse_score, conf.percentile_cleanse)
            mask = cleanse_score <= thres
            self.tr_f = self.tr_f[mask]

        self.te_f = featurebank.get(dataset_name, feature_name, 'test', uvadmode)

        self.grader = None

    def get(self):
        gr = grader.KNNGrader(self.tr_f, K=self.conf.NN, key='app')
        tr_scorebbox = gr.grade_flat(self.tr_f)
        minv, maxv = calc_minmax(tr_scorebbox)
        te_scorebbox = gr.grade(self.te_f)
        return normalize(te_scorebbox, minv, maxv)
