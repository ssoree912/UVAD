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
        self.tr_x_dim = tr_x.shape[1]
        self.key = key
        # CPU 배열을 연속/float32로 보관 (fallback용)
        self.tr_x_cpu = np.ascontiguousarray(tr_x.astype(np.float32), dtype=np.float32)
        self.use_gpu = False

        with task('Building KNN index', debug=True):
            try:
                # Force GPU memory cleanup before creating new resources
                import gc
                gc.collect()
                
                # Try Multi-GPU sharding first
                ngpu = faiss.get_num_gpus()
                if ngpu > 1:
                    logging.info(f"[{key}] Trying Multi-GPU sharding with {ngpu} GPUs")
                    try:
                        # Create resources for all GPUs
                        self.res = [faiss.StandardGpuResources() for _ in range(ngpu)]
                        for r in self.res:
                            r.setTempMemory(512 * 1024 * 1024)
                        
                        # Create CPU index first
                        cpu_index = faiss.IndexFlatL2(self.tr_x_dim)
                        cpu_index.add(self.tr_x_cpu)
                        
                        # Multi-GPU cloning with sharding
                        co = faiss.GpuMultipleClonerOptions()
                        co.shard = True         # DB 샤딩 (데이터 분산)
                        co.useFloat16 = False   # Float32 유지
                        
                        # Clone to multiple GPUs
                        self.index = faiss.index_cpu_to_gpu_multiple(
                            self.res, list(range(ngpu)), cpu_index, co
                        )
                        self.use_gpu = True
                        
                        logging.info(f"[{key}] Successfully created Multi-GPU sharded index on {ngpu} GPUs with {len(tr_x):,} vectors")
                        
                    except Exception as multi_gpu_error:
                        logging.warning(f"[{key}] Multi-GPU failed: {multi_gpu_error}, falling back to single GPU")
                        # Clean up multi-GPU resources
                        if hasattr(self, 'res'):
                            del self.res
                        # Fall through to single GPU
                        raise RuntimeError("Multi-GPU failed")
                
                else:
                    # Single GPU fallback
                    logging.info(f"[{key}] Using single GPU (only {ngpu} GPU available)")
                    raise RuntimeError("Single GPU mode")
                    
            except RuntimeError:
                # Single GPU fallback
                try:
                    self.res = faiss.StandardGpuResources()
                    
                    # Relax temp memory limit (512MB)
                    temp_memory_mb = 512
                    if hasattr(self.res, 'setTempMemory'):
                        self.res.setTempMemory(temp_memory_mb * 1024 * 1024)
                    
                    # Use float32 with proper config
                    cfg = faiss.GpuIndexFlatConfig()
                    cfg.device = 0
                    cfg.useFloat16 = False
                    
                    self.index = faiss.GpuIndexFlatL2(self.res, self.tr_x_dim, cfg)
                    self.index.add(self.tr_x_cpu)
                    self.use_gpu = True
                    
                    logging.info(f"[{key}] Successfully created single GPU index with {len(tr_x):,} vectors")
                    
                except RuntimeError as e:
                    if 'cudaMalloc' in str(e) or 'out of memory' in str(e) or 'CUBLAS' in str(e):
                        logging.warning(
                            "[%s] GPU completely failed (%s); falling back to CPU Faiss index.",
                            key, e
                        )
                        # Clean up failed GPU resources
                        if hasattr(self, 'res') and self.res is not None:
                            if isinstance(self.res, list):
                                for r in self.res:
                                    del r
                            else:
                                del self.res
                        self.res = None
                        
                        # Create CPU index
                        self.index = faiss.IndexFlatL2(self.tr_x_dim)
                        self.index.add(self.tr_x_cpu)
                        self.use_gpu = False
                        logging.info(f"[{key}] Successfully created CPU index with {len(tr_x):,} vectors")
                    else:
                        raise
    
    def __del__(self):
        """Clean up GPU resources when grader is deleted"""
        try:
            if hasattr(self, 'res') and self.res is not None:
                if isinstance(self.res, list):
                    for r in self.res:
                        del r
                else:
                    del self.res
            if hasattr(self, 'index') and self.index is not None:
                del self.index
        except:
            pass

    def grade_flat(self, te_x):  # [M, D]
        te_x = np.ascontiguousarray(te_x, dtype=np.float32)  # 연속/float32 보장

        B = 256  # 배치 크기는 256 권장 (128~512 범위)
        ALIGN = 32  # 32의 배수로 패딩
        all_distances = []

        for i in range(0, len(te_x), B):
            batch = te_x[i:i + B]
            m = len(batch)
            
            # 32의 배수로 패딩 (마지막 배치 포함)
            pad = (-m) % ALIGN
            if pad:
                pad_block = np.repeat(batch[-1:], pad, axis=0)
                batch_padded = np.vstack([batch, pad_block])
            else:
                batch_padded = batch

            try:
                # 반드시 연속 메모리로 보장
                batch_padded = np.ascontiguousarray(batch_padded, dtype=np.float32)
                Ds, _ = self.index.search(batch_padded, self.K + 1)
                Ds = Ds[:m]  # 패딩 자르기
                all_distances.append(Ds)
                
            except RuntimeError as e:
                if 'CUBLAS' in str(e) or 'cuBLAS' in str(e):
                    logging.warning(f"[{self.key}] cuBLAS fail at i={i}, falling back to CPU for this chunk")
                    
                    # CPU 인덱스 준비 (1회만)
                    if not hasattr(self, '_cpu_index'):
                        self._cpu_index = faiss.IndexFlatL2(self.tr_x_dim)
                        self._cpu_index.add(self.tr_x_cpu)
                        logging.info(f"[{self.key}] Created emergency CPU index")
                    
                    # 패딩 없이 원본 배치를 CPU로 처리
                    Ds_cpu, _ = self._cpu_index.search(batch, self.K + 1)
                    all_distances.append(Ds_cpu)
                    
                elif 'cuda' in str(e).lower():
                    # GPU 완전 실패 시 영구 CPU로 전환
                    logging.warning(f"[{self.key}] GPU completely failed, switching to CPU permanently")
                    self.index = faiss.IndexFlatL2(self.tr_x_dim)
                    self.index.add(self.tr_x_cpu)
                    self.use_gpu = False
                    
                    # 현재 배치부터 CPU로 계속 처리
                    Ds_cpu, _ = self.index.search(batch, self.K + 1)
                    all_distances.append(Ds_cpu)
                else:
                    raise

        # 모든 배치 결과 연결
        Ds = np.concatenate(all_distances, axis=0)
        
        ret = []
        for vs in Ds:
            if np.any(vs == 0):
                vs[np.where(vs == 0)[0][0]] = np.inf  # remove only one exact match.
            ret.append(vs[np.argsort(vs)][:-1])
        Ds = np.asarray(ret)
        return np.mean(Ds, axis=1)
