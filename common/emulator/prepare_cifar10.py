"""Decode the HF-mirror CIFAR-10 parquet files (downloaded because the
canonical https://www.cs.toronto.edu/~kriz/... host and the usual S3/GCS
mirrors torchvision falls back to were all unreachable from this machine)
into plain uint8 numpy arrays, once, so every later script just does
np.load() instead of re-decoding 60k PNGs per run.
"""
import io
import os

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

DATA_DIR = os.path.dirname(os.path.abspath(__file__)).replace("/scripts", "/data")


def decode_split(parquet_path, out_prefix):
    table = pq.ParquetFile(parquet_path).read()
    rows = table.to_pylist()
    n = len(rows)
    images = np.empty((n, 32, 32, 3), dtype=np.uint8)
    labels = np.empty((n,), dtype=np.int64)
    for i, row in enumerate(rows):
        img = Image.open(io.BytesIO(row["img"]["bytes"])).convert("RGB")
        images[i] = np.array(img, dtype=np.uint8)
        labels[i] = row["label"]
    np.save(f"{DATA_DIR}/{out_prefix}_images.npy", images)
    np.save(f"{DATA_DIR}/{out_prefix}_labels.npy", labels)
    print(f"{out_prefix}: {n} images -> {out_prefix}_images.npy / {out_prefix}_labels.npy")


if __name__ == "__main__":
    decode_split(f"{DATA_DIR}/cifar10_train.parquet", "cifar10_train")
    decode_split(f"{DATA_DIR}/cifar10_test.parquet", "cifar10_test")
