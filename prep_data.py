"""Builds data/c10_train.npz, data/c10_test.npz and data/c100_test.npz (uint8 images, int64 labels).

Default: download CIFAR-10 and CIFAR-100 with torchvision.
Alternative: pass a directory holding the HuggingFace parquet files
(cifar10_train.parquet, cifar10_test.parquet, cifar100_test.parquet) as the first argument.
"""
import os, sys, io
import numpy as np
os.makedirs('data', exist_ok=True)

if len(sys.argv) > 1:
    import pandas as pd
    from PIL import Image
    D = sys.argv[1].rstrip('/') + '/'
    def conv(f, lab):
        d = pd.read_parquet(D + f)
        X = np.stack([np.array(Image.open(io.BytesIO(b['bytes'])).convert('RGB')) for b in d['img']])
        return X, d[lab].to_numpy().astype(np.int64)
    for f, lab, o in [('cifar10_train.parquet', 'label', 'c10_train'), ('cifar10_test.parquet', 'label', 'c10_test'), ('cifar100_test.parquet', 'fine_label', 'c100_test')]:
        X, y = conv(f, lab); np.savez(f'data/{o}.npz', X=X, y=y); print(o, X.shape)
else:
    import torchvision
    for ds, train, o in [(torchvision.datasets.CIFAR10, True, 'c10_train'), (torchvision.datasets.CIFAR10, False, 'c10_test'), (torchvision.datasets.CIFAR100, False, 'c100_test')]:
        d = ds('data/_raw', train=train, download=True)
        X = np.asarray(d.data, dtype=np.uint8); y = np.asarray(d.targets, dtype=np.int64)
        np.savez(f'data/{o}.npz', X=X, y=y); print(o, X.shape)
