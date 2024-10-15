import numpy as np

__all__ = ['get']


def get(dataset_name, feature_name, mode, uvadmode='partial'):
    assert mode in ['train', 'test']
    assert uvadmode in ['partial', 'merge']

    if uvadmode == 'merge':
        return _get_mergemode(dataset_name, feature_name, mode)
    elif uvadmode == 'partial':
        return _get_partialmode(dataset_name, feature_name, mode)
    else:
        raise ValueError()


def _get_mergemode(dataset_name, feature_name, mode):
    dpath = f'features/{dataset_name}'
    if mode == 'train':
        fpath1 = f'{dpath}/train/{feature_name}.npy'
        fpath2 = f'{dpath}/test/{feature_name}.npy'

        d1 = np.load(fpath1, allow_pickle=True)
        d2 = np.load(fpath2, allow_pickle=True)
        d1 = np.concatenate(d1, 0)
        d2 = [v for v in d2 if len(v)]
        d2 = np.concatenate(d2, 0)
        return np.concatenate([d1, d2])

    elif mode == 'test':
        fpath2 = f'{dpath}/test/{feature_name}.npy'
        d2 = np.load(fpath2, allow_pickle=True)
        return d2

    else:
        raise ValueError()


def _get_partialmode(dataset_name, feature_name, mode):
    dpath = f'features/{dataset_name}'
    if mode == 'train':
        fpath1 = f'{dpath}/test/{feature_name}.npy'

        d1 = np.load(fpath1, allow_pickle=True)
        d1 = np.concatenate(d1, 0)
        return d1

    elif mode == 'test':
        fpath2 = f'{dpath}/test/{feature_name}.npy'
        d2 = np.load(fpath2, allow_pickle=True)
        return d2

    else:
        raise ValueError()
