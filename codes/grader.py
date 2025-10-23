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
    def __init__(self, tr_x, K=1, key=''):
        super().__init__(key)
        self.K = K
        self.tr_x = tr_x
        self.use_gpu = False

        with task('Building KNN index', debug=True):
            try:
                # Force GPU memory cleanup before creating new resources
                import gc
                gc.collect()
                
                # Create GPU resources with strict memory limit
                self.res = faiss.StandardGpuResources()
                
                # Set temp memory limit to prevent excessive allocation
                temp_memory_mb = 128  # Even stricter: 128MB limit
                if hasattr(self.res, 'setTempMemory'):
                    self.res.setTempMemory(temp_memory_mb * 1024 * 1024)
                
                # Use float16 configuration to halve memory usage
                cfg = faiss.GpuIndexFlatConfig()
                cfg.device = 0
                cfg.useFloat16 = True  # CRITICAL: Half precision for 50% memory savings
                
                self.index = faiss.GpuIndexFlatL2(self.res, tr_x.shape[1], cfg)
                self.index.add(tr_x.astype(np.float32))
                self.use_gpu = True
                
                logging.info(f"[{key}] Successfully created GPU index with {len(tr_x):,} vectors")
                
            except RuntimeError as e:
                if 'cudaMalloc' in str(e) or 'out of memory' in str(e):
                    logging.warning(
                        "[%s] GPU allocation failed (%s); falling back to CPU Faiss index.",
                        key, e
                    )
                    # Clean up failed GPU resources
                    if hasattr(self, 'res') and self.res is not None:
                        del self.res
                    self.res = None
                    
                    # Create CPU index
                    self.index = faiss.IndexFlatL2(tr_x.shape[1])
                    self.index.add(tr_x.astype(np.float32))
                    self.use_gpu = False
                    logging.info(f"[{key}] Successfully created CPU index with {len(tr_x):,} vectors")
                else:
                    raise
    
    def __del__(self):
        """Clean up GPU resources when grader is deleted"""
        try:
            if hasattr(self, 'res') and self.res is not None:
                del self.res
            if hasattr(self, 'index') and self.index is not None:
                del self.index
        except:
            pass

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
