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
                        self.res_list = [faiss.StandardGpuResources() for _ in self.gpu_ids]
                        for r in self.res_list:
                            if hasattr(r, 'setTempMemory'):
                                r.setTempMemory(temp_memory_mb * 1024 * 1024)
                        co = faiss.GpuMultipleClonerOptions()
                        co.shard = True
                        co.useFloat16 = False
                        self.index = faiss.index_cpu_to_gpu_multiple_py(
                            self.res_list,
                            [int(g) for g in self.gpu_ids],
                            cpu_index,
                            co
                        )
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
                    self.res_list = None
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
