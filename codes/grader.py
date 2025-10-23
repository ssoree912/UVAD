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
                
                # Use float32 (float16 causes CUBLAS errors on some matrix sizes)
                cfg = faiss.GpuIndexFlatConfig()
                cfg.device = 0
                cfg.useFloat16 = False  # Disable float16 due to CUBLAS issues
                
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
        te_x = te_x.astype(np.float32)
        
        # Batch search to avoid large GEMM operations that cause CUBLAS errors
        batch_size = 32  # Small batches to avoid CUBLAS issues
        all_distances = []
        
        for i in range(0, len(te_x), batch_size):
            batch = te_x[i:i + batch_size]
            try:
                Ds, _ = self.index.search(batch, self.K + 1)
                all_distances.append(Ds)
            except RuntimeError as e:
                if 'CUBLAS' in str(e) or 'cuBLAS' in str(e):
                    logging.warning(f"[{self.key}] CUBLAS error on batch {i}, retrying with smaller batch")
                    # Retry with even smaller batches
                    for j in range(i, min(i + batch_size, len(te_x))):
                        single_query = te_x[j:j+1]
                        Ds_single, _ = self.index.search(single_query, self.K + 1)
                        all_distances.append(Ds_single)
                else:
                    raise
        
        # Concatenate all batch results
        Ds = np.concatenate(all_distances, axis=0)
        
        ret = []
        for vs in Ds:
            if np.any(vs == 0):
                vs[np.where(vs == 0)[0][0]] = np.inf  # remove only one exact match.
            ret.append(vs[np.argsort(vs)][:-1])
        Ds = np.asarray(ret)
        return np.mean(Ds, axis=1)
