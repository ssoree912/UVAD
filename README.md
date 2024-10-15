# Cleansed k-nearest neighbor (CKNN)

This is the official PyTorch implementation of CKNN, published in **CIKM 2024** ([paper](https://arxiv.org/abs/2408.03014)).

## Citation

If you use code or refer to CKNN for your research, please cite your paper:
```
@article{yi2024cknn,
  title={CKNN: Cleansed k-Nearest Neighbor for Unsupervised Video Anomaly Detection},
  author={Yi, Jihun and Yoon, Sungroh},
  journal={arXiv preprint arXiv:2408.03014},
  year={2024}
}
```

# Installation

#### Step 1. Install libraries
- We used python=3.7
- Install libraries in [requirements.txt](requirements.txt).

#### Step 2. Download files

- [Download features](https://drive.google.com/file/d/1FT97l_fN6rvvXYRvEnq4SKoIOyP8RNOK/view?usp=sharing)
- [Download meta data](https://drive.google.com/file/d/1BmoY_BQnXxMnS8etydHMaqXg13c3uJ7l/view?usp=sharing)
- [Download all 6 files of patches in this folder](https://drive.google.com/drive/folders/1PK7-0K-it4Ldt-uSYNtCbj1-TKzafYBi?usp=sharing)

#### Step 3. Untar all *.tar.gz files and place properly
- File structure should be like below.
```
AnonymousCKNN
├── features
│   ├── avenue
│   │   ├── test
│   │   └── train
│   ├── ped2
│   │   ├── test
│   │   └── train
│   └── shanghaitech
│       ├── test
│       └── train
├── meta
│   ├── frame_labels_avenue.npy
│   ├── frame_labels_ped2.npy
│   ├── frame_labels_shanghaitech.npy
│   ├── test_bboxes_avenue.npy
│   ├── test_bboxes_ped2.npy
│   ├── test_bboxes_shanghaitech.npy
│   ├── test_lengths_avenue.npy
│   ├── test_lengths_ped2.npy
│   ├── test_lengths_shanghaitech.npy
│   ├── train_bboxes_avenue.npy
│   ├── train_bboxes_ped2.npy
│   └── train_bboxes_shanghaitech.npy
├── patches
│   ├── avenue
│   │   ├── test
│   │   └── train
│   ├── ped2
│   │   ├── test
│   │   └── train
│   └── shanghaitech
│       ├── test
│       └── train
...
``` 

#### Step 4. Generate pseudo anomaly scores

```
./run1_pseudo_anomaly_scores.sh avenue
```

#### Step 5. Evaluate on each dataset

```
./run2_evaluate.sh avenue
```
