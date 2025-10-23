from abc import abstractmethod
import logging

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
    def __init__(self, tr_x, K=1, key='', gpu_ids=None, temp_memory_mb: int = 512):
        super().__init__(key)
        self.K = K
        self.dim = tr_x.shape[1]
        self.key = key
        self.gpu_ids = list(gpu_ids) if gpu_ids else []
        self.use_gpu = False
        self.multi_gpu = len(self.gpu_ids) > 1
        self.res = None
        self.res_list = None

        features = np.ascontiguousarray(tr_x.astype(np.float32))
        cpu_index = faiss.IndexFlatL2(self.dim)
        cpu_index.add(features)

        with task('Building KNN index', debug=True):
            if self.gpu_ids:
                try:
                    if self.multi_gpu:
                        self.res_list = []
                        res_vec = faiss.GpuResourcesVector()
                        dev_vec = faiss.IntVector()
                        for gid in self.gpu_ids:
                            res = faiss.StandardGpuResources()
                            if hasattr(res, 'setTempMemory'):
                                res.setTempMemory(temp_memory_mb * 1024 * 1024)
                            self.res_list.append(res)
                            res_vec.push_back(res)
                            dev_vec.push_back(int(gid))
                        co = faiss.GpuMultipleClonerOptions()
                        co.shard = True
                        co.useFloat16 = False
                        self.index = faiss.index_cpu_to_gpu_multiple(res_vec, dev_vec, cpu_index, co)
                    else:
                        gpu_id = int(self.gpu_ids[0])
                        self.res = faiss.StandardGpuResources()
                        if hasattr(self.res, 'setTempMemory'):
                            self.res.setTempMemory(temp_memory_mb * 1024 * 1024)
                        self.index = faiss.index_cpu_to_gpu(self.res, gpu_id, cpu_index)
                    self.use_gpu = True
                    cpu_index = None
                    logging.info(
                        "[%s] Created %sGPU Faiss index on GPUs %s with %s vectors",
                        key,
                        "multi-" if self.multi_gpu else "",
                        self.gpu_ids,
                        f"{len(tr_x):,}"
                    )
                except Exception as e:
                    logging.warning(
                        "[%s] GPU setup failed (%s); falling back to CPU index.",
                        key, e
                    )
                    self.res = None
                    self.res_list = []
                    self.use_gpu = False
                    self.multi_gpu = False
                    self.index = faiss.IndexFlatL2(self.dim)
                    self.index.add(features)
            else:
                self.index = cpu_index

    def __del__(self):
        try:
            if isinstance(self.res_list, list):
                for r in self.res_list:
                    del r
            if self.res is not None:
                del self.res
        except Exception:
            pass

    def grade_flat(self, te_x):  # [M, D]
        te_x = np.ascontiguousarray(te_x.astype(np.float32))
        Ds, _ = self.index.search(te_x, self.K + 1)
        
        ret = []
        for vs in Ds:
            if np.any(vs == 0):
                vs[np.where(vs == 0)[0][0]] = np.inf  # remove only one exact match.
            ret.append(vs[np.argsort(vs)][:-1])
        Ds = np.asarray(ret)
        return np.mean(Ds, axis=1)
