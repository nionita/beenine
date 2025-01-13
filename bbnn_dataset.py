# vim: ts=4 sw=4 et

import os.path
import pandas as pd
import glob

from torch.utils.data import Dataset

class BBNNDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

        # Keep all file names to load them when needed
        self.feature_files = self.filenames('*-feat.txt')
        self.target_files  = self.filenames('*-targ.txt')

        # Also keep file lenghts to calculate the relativ idx
        self.file_recs  = []

        assert len(self.target_files) != 0

        self.len_ = None
        self.ds_features = None
        self.ds_targets  = None
        self.cur_idx     = None
        self.cur_file    = None
        self.cur_base    = None

    def filenames(self, pat):
        found = []
        for file_name in glob.glob(pat, root_dir=self.data_dir):
            abs_name = os.path.join(self.data_dir, file_name)
            found.append(abs_name)
        return found

    def __len__(self):
        if self.len_ is None:
            self.len_ = 0
            for file_name in self.target_files:
                ds = pd.read_csv(file_name, header=None)
                self.len_ += ds.shape[0]
                self.file_recs.append(ds.shape[0])
        return self.len_

    def __getitem__(self, idx):
        last_file = self.cur_file
        rel_idx = idx
        self.cur_base = 0
        found = False
        for i, recs in enumerate(self.file_recs):
            if rel_idx < recs:
                self.cur_file = i
                found = True
                break
            self.cur_base += recs
            rel_idx       -= recs
        assert found

        if last_file is None or self.cur_file != last_file:
            self.ds_features = pd.read_csv(self.feature_files[self.cur_file], header=None)
            self.ds_targets  = pd.read_csv(self.target_files[self.cur_file], header=None)
        features = self.ds_features.iloc[rel_idx, :]
        target   = self.ds_targets.iloc[rel_idx, [0]] # only score for now

        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            target = self.target_transform(target)

        return features, target
