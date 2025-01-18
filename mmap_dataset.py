# Mem mapped dataset inspired from https://lukesalamone.github.io/posts/very-large-datasets/

# vim: ts=4 sw=4 et

import sys
import os
import os.path
import numpy as np
import glob
import re
import dask.array as da
import torch
from torch.utils.data import Dataset


class MemmapDataset(Dataset):
    def __init__(self, ds_dir):
        self.features = []
        self.targets = []

        roco = re.compile(r'.*-(\d+)x(\d+)\.memmap$')

        feat_pat = os.path.join(ds_dir, '*-feat-*.memmap')
        for file_name in glob.glob(feat_pat):
            match = roco.match(file_name)
            if match:
                shape = (int(match.group(1)), int(match.group(2)))
                self.features.append(np.memmap(file_name, dtype=np.float32, mode='r', shape=shape))
            else:
                print(f'Bad file {file_name}: skip')

        targ_pat = os.path.join(ds_dir, '*-targ-*.memmap')
        for file_name in glob.glob(targ_pat):
            match = roco.match(file_name)
            if match:
                shape = (int(match.group(1)), int(match.group(2)))
                self.targets.append(np.memmap(file_name, dtype=np.float32, mode='r', shape=shape))
            else:
                print(f'Bad file {file_name}: skip')

        self.features = da.concatenate(self.features, axis=0)
        self.targets = da.concatenate(self.targets, axis=0)

        print(f'MM: {self.features.shape}')
        print(f'MM: {self.targets.shape}')

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return (
                torch.tensor(self.features[idx].compute(), dtype=torch.float32),
                torch.tensor(self.targets[idx, 0].compute(), dtype=torch.float32)
                )

def make_memmap_file(file_path, out_dir, dtype=np.float32):
    print(f'+ make_mem_file from {file_path}')
    file_name = os.path.basename(file_path)
    base, _ = os.path.splitext(file_name)

    lines = []
    with open(file_path, 'r') as f:
        for line in f:
            lines.append(np.fromstring(line, dtype=dtype, sep=','))

    rows = len(lines)
    cols = len(lines[0])

    new_file_name = f'{base}-{rows}x{cols}.memmap'
    new_file_path = os.path.join(out_dir, new_file_name)
    print(f'--> {new_file_path}')

    # create the mem mapped file on the disk:
    memmap_file = np.memmap(os.path.join(out_dir, new_file_name),
            dtype=dtype, mode='w+', shape=(rows, cols))
    for i, arr in enumerate(lines):
        memmap_file[i, :] = arr

    memmap_file.flush()

if __name__ == '__main__':
    input_dir = sys.argv[1]
    out_dir   = sys.argv[2] if len(sys.argv) > 2 else input_dir

    # Feature files:
    ipat = os.path.join(input_dir, '*-feat.txt')
    for path in glob.glob(ipat):
        make_memmap_file(path, out_dir, dtype=np.float32)

    # Target files:
    ipat = os.path.join(input_dir, '*-targ.txt')
    for path in glob.glob(ipat):
        make_memmap_file(path, out_dir, dtype=np.float32)
