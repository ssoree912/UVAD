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
    def __init__(self, tr_x, K=1, key='', gpu_ids=None, temp_memory_mb: int = 64, use_ivf: bool = False, ivf_nlist: int = 4096, ivf_nprobe: int = 32, use_float16: bool = False):
        super().__init__(key)
        self.K = K
        self.dim = tr_x.shape[1]
        self.key = key
        self.gpu_ids = list(gpu_ids) if gpu_ids else []
        self.use_gpu = False
        self.multi_gpu = len(self.gpu_ids) > 1
        self.res = None
        self.res_list = None
        self.use_ivf = use_ivf
        self.ivf_nlist = ivf_nlist
        self.ivf_nprobe = ivf_nprobe
        self.use_float16 = use_float16
        self.temp_memory_mb = temp_memory_mb

        features = np.ascontiguousarray(tr_x.astype(np.float32))
        # Store original features for emergency CPU fallback
        self._original_features = features.copy()
        
        # Build CPU index (Flat L2 or IVF-Flat)
        if self.use_ivf:
            cpu_index = self._build_ivf_index(features)
        else:
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
                                res.setTempMemory(self.temp_memory_mb * 1024 * 1024)
                            self.res_list.append(res)
                            res_vec.push_back(res)
                            dev_vec.push_back(int(gid))
                        co = faiss.GpuMultipleClonerOptions()
                        co.shard = True
                        co.useFloat16 = self.use_float16
                        self.index = faiss.index_cpu_to_gpu_multiple(res_vec, dev_vec, cpu_index, co)
                    else:
                        gpu_id = int(self.gpu_ids[0])
                        self.res = faiss.StandardGpuResources()
                        if hasattr(self.res, 'setTempMemory'):
                            self.res.setTempMemory(self.temp_memory_mb * 1024 * 1024)
                        self.index = faiss.index_cpu_to_gpu(self.res, gpu_id, cpu_index)
                    self.use_gpu = True
                    cpu_index = None
                    index_type = "IVF-Flat" if self.use_ivf else "Flat-L2"
                    float_type = "float16" if self.use_float16 else "float32"
                    logging.info(
                        "[%s] Created %sGPU Faiss %s index (%s) on GPUs %s with %s vectors, temp_mem=%dMB",
                        key,
                        "multi-" if self.multi_gpu else "",
                        index_type,
                        float_type,
                        self.gpu_ids,
                        f"{len(tr_x):,}",
                        self.temp_memory_mb
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

    def _build_ivf_index(self, features):
        """Build IVF-Flat index for memory efficiency"""
        n_samples = len(features)
        
        # Adjust nlist based on data size
        nlist = min(self.ivf_nlist, int(np.sqrt(n_samples)))
        nlist = max(nlist, 1)  # Ensure at least 1 cluster
        
        # Create IVF index
        quantizer = faiss.IndexFlatL2(self.dim)
        index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_L2)
        
        # Train with subset if data is large
        train_size = min(len(features), 100000)
        train_subset = features[:train_size] if train_size < len(features) else features
        index.train(train_subset)
        
        # Add all features
        index.add(features)
        
        # Set search parameters
        index.nprobe = min(self.ivf_nprobe, nlist)
        
        logging.info(f"[{self.key}] Built IVF index with nlist={nlist}, nprobe={index.nprobe}")
        return index
    
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
        
        # Use small batches to avoid cuBLAS crashes
        batch_size = 16  # Very small batches to prevent cuBLAS errors
        all_distances = []
        
        for i in range(0, len(te_x), batch_size):
            batch = te_x[i:i + batch_size]
            
            try:
                # Ensure contiguous memory for batch
                batch = np.ascontiguousarray(batch, dtype=np.float32)
                Ds, _ = self.index.search(batch, self.K + 1)
                all_distances.append(Ds)
                
            except RuntimeError as e:
                if 'CUBLAS' in str(e) or 'cuBLAS' in str(e):
                    logging.warning(f"[{self.key}] cuBLAS fail at batch {i//batch_size}, falling back to CPU for this batch")
                    
                    # Create emergency CPU index if not exists
                    if not hasattr(self, '_emergency_cpu_index'):
                        self._emergency_cpu_index = faiss.IndexFlatL2(self.dim)
                        # Reconstruct training data from GPU index if possible
                        if hasattr(self, '_original_features'):
                            self._emergency_cpu_index.add(self._original_features)
                        else:
                            logging.error(f"[{self.key}] No training data available for CPU fallback")
                            raise
                    
                    # Process batch with CPU
                    Ds_cpu, _ = self._emergency_cpu_index.search(batch, self.K + 1)
                    all_distances.append(Ds_cpu)
                    
                elif 'cuda' in str(e).lower() or 'gpu' in str(e).lower():
                    # Complete GPU failure - switch to CPU permanently
                    logging.warning(f"[{self.key}] GPU failed completely, switching to CPU mode")
                    
                    if not hasattr(self, '_emergency_cpu_index'):
                        self._emergency_cpu_index = faiss.IndexFlatL2(self.dim)
                        if hasattr(self, '_original_features'):
                            self._emergency_cpu_index.add(self._original_features)
                        else:
                            logging.error(f"[{self.key}] No training data for CPU fallback")
                            raise
                    
                    # Replace index with CPU version
                    self.index = self._emergency_cpu_index
                    self.use_gpu = False
                    
                    # Process remaining batches with CPU
                    remaining = te_x[i:]
                    Ds_remaining, _ = self.index.search(remaining, self.K + 1)
                    all_distances.append(Ds_remaining)
                    break
                else:
                    raise
        
        # Concatenate all results
        Ds = np.concatenate(all_distances, axis=0) if all_distances else np.array([])
        
        ret = []
        for vs in Ds:
            if np.any(vs == 0):
                vs[np.where(vs == 0)[0][0]] = np.inf  # remove only one exact match.
            ret.append(vs[np.argsort(vs)][:-1])
        Ds = np.asarray(ret)
        return np.mean(Ds, axis=1)
