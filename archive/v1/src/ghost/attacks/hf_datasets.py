"""Fast CIFAR-10/100 loader backed by the HuggingFace parquet mirror.

The canonical torchvision download host (cs.toronto.edu) is frequently
throttled to a crawl from some networks. This loader fetches the same
images from HF's CDN instead and exposes a torch Dataset with the same
(img, label) contract used by DataLoader/Subset throughout the attack
suite, so it is a drop-in replacement for torchvision.datasets.CIFAR10/100
wherever those are used with the default plain_text/fine_label configs.
"""

import io
import os
import urllib.request

import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset

_HF_URLS = {
    "cifar10": {
        True: "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet",
        False: "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet",
    },
    "cifar100": {
        True: "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/train-00000-of-00001.parquet",
        False: "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/test-00000-of-00001.parquet",
    },
}
_IMG_COL = {"cifar10": "img", "cifar100": "img"}
_LABEL_COL = {"cifar10": "label", "cifar100": "fine_label"}


def _cache_path(root, name, train):
    split = "train" if train else "test"
    return os.path.join(root, f"{name}_{split}.parquet")


def _download(url, dest):
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)


class HFImageDataset(Dataset):
    """cifar10 / cifar100 test or train split, downloaded once and cached as parquet."""

    def __init__(self, name, root, train, transform=None):
        assert name in _HF_URLS, f"unsupported dataset {name}"
        path = _cache_path(root, name, train)
        _download(_HF_URLS[name][train], path)
        table = pq.read_table(path)
        self._img_bytes = [d["bytes"] for d in table.column(_IMG_COL[name]).to_pylist()]
        self._labels = table.column(_LABEL_COL[name]).to_pylist()
        self.transform = transform

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        img = Image.open(io.BytesIO(self._img_bytes[idx])).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self._labels[idx]
