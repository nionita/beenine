# vim: ts=4 sw=4 et

import os.path
import numpy as np
#import pandas as pd
import glob

from torch.utils.data import IterableDataset

FEATURE_LEN = 384 * 2

class BBNNDataset(IterableDataset):
    def __init__(self, data_dir):
        super(BBNNDataset).__init__()

        # Keep all file names to load them when needed
        # We should check that the file names are identical except -feat and -targ!
        self.feat_files = self.filenames(data_dir, '*-feat.txt')
        self.targ_files = self.filenames(data_dir, '*-targ.txt')

        assert len(self.feat_files) != 0
        assert len(self.targ_files) != 0
        assert len(self.feat_files) == len(self.targ_files)

        self.ds_feats = None
        self.ds_targs = None
        self.cur_idx  = None
        self.cur_file = None

    def filenames(self, data_dir, pat):
        found = []
        #print(f'filenames in {data_dir} with {pat}')
        for file_name in glob.glob(pat, root_dir=data_dir):
            abs_name = os.path.join(data_dir, file_name)
            found.append(abs_name)
        return found

    def read_file(self, file_name):
        print(f'DS: read new data file {file_name}')
        arrs = []
        with open(file_name, 'r') as f:
            for line in f:
                arrs.append(np.fromstring(line, dtype=int, sep=','))
        print(f'DS: got {len(arrs)} records')
        return arrs

    def __next__(self):
        #print('DS: __next__ call')
        new_file = False
        if self.ds_feats is None:
            #print('DS: __next__ 1')
            new_file = True
            self.cur_file = 0
        elif self.cur_idx >= len(self.ds_feats):
            #print('DS: __next__ 2')
            new_file = True
            self.cur_file += 1
            if self.cur_file >= len(self.feat_files):
                #print('DS: __next__ 3')
                self.cur_file = 0
        if new_file:
            #print('DS: __next__ 4')
            self.cur_idx = 0
            self.ds_feats = None
            self.ds_targs = None
            self.ds_feats = self.read_file(self.feat_files[self.cur_file])
            self.ds_targs = self.read_file(self.targ_files[self.cur_file])

        # Yield the current record and increase current index
        # We must set 1 on the feature indices
        feix = self.ds_feats[self.cur_idx]
        feat = np.zeros(FEATURE_LEN, dtype=np.float32)
        feat[feix] = 1.0
        targ = self.ds_targs[self.cur_idx][0].astype(np.float32) # only score for now
        self.cur_idx += 1
        #print('DS: __next__ 5')
        return feat, targ

    def __iter__(self):
        return self

if __name__ == '__main__':
    print('Test dataset')
    train_dir = 'E:\\extract\\2025\\test-1\\train'
    tds = BBNNDataset(train_dir)
    i = 0
    for X, y in tds:
        print(f'{X} -> {y}')
        #print(f'Shapes: {X.shape} {X.type} {y.shape} {y.type}')
        i += 1
        if i > 10:
            break
