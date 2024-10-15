from abc import abstractmethod

import faiss
import numpy as np
from tqdm import tqdm

from .utils import task


class ABCGrader:
    def __init__(self, key=''):
        self.key = key

    @abstractmethod
    def grade_flat(self, te_x):  # [M, D]
        return NotImplemented

    def grade(self, te_x):
        ret = []
        T = len(te_x)
        for t in tqdm(range(T), total=T, desc=f'Grading bboxes {self.key}', disable=False, leave=False):
            feats = te_x[t]
            if isinstance(feats, list):
                feats = np.asarray(feats)
            if feats.shape[0]:
                ret.append(self.grade_flat(feats))
            else:
                ret.append(np.asarray([], dtype=np.float32))
        return np.asarray(ret, dtype=object)


class KNNGrader(ABCGrader):
    def __init__(self, tr_x, K=1, key=''):
        super().__init__(key)
        self.K = K
        self.tr_x = tr_x

        with task('Building KNN index', debug=True):
            self.res = faiss.StandardGpuResources()
            self.index = faiss.IndexFlatL2(tr_x.shape[1])
            self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index)
            self.index.add(tr_x.astype(np.float32))

    def grade_flat(self, te_x):  # [M, D]
        Ds, _ = self.index.search(te_x.astype(np.float32), self.K + 1)
        ret = []
        for vs in Ds:
            if np.any(vs == 0):
                vs[np.where(vs == 0)[0][0]] = np.inf  # remove only one exact match.
            ret.append(vs[np.argsort(vs)][:-1])
        Ds = np.asarray(ret)
        # print(Ds.shape)
        return np.mean(Ds, axis=1)
