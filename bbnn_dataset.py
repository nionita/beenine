# vim: ts=4 sw=4 et

import os.path
import numpy as np
import glob
import torch

from torch.utils.data import IterableDataset, get_worker_info

FEATURE_LEN = 384 * 2

class BBNNDataset(IterableDataset):
    def __init__(self, data_dir, rezult_target=False, evaluate=False, skip=0):
        super(BBNNDataset).__init__()

        self.rezult_target = rezult_target  # default: score
        self.evaluate = evaluate
        self.skip = skip

        if self.evaluate:
            self.feat_files = [ data_dir ]  # this is actuall a file
            self.targ_files = []    # not needed for evaluation
        else:
            # Keep all file names to load them when needed
            # We should check that the file names are identical except -feat and -targ!
            self.feat_files = self.filenames(data_dir, '*-feat.txt')
            self.targ_files = self.filenames(data_dir, '*-targ.txt')

            assert len(self.feat_files) != 0
            assert len(self.targ_files) != 0
            assert len(self.feat_files) == len(self.targ_files)

    def filenames(self, data_dir, pat):
        found = []
        #print(f'filenames in {data_dir} with {pat}')
        for file_name in glob.glob(pat, root_dir=data_dir):
            abs_name = os.path.join(data_dir, file_name)
            found.append(abs_name)
        return found

    #    targ = self.ds_targs[self.cur_idx][0].astype(np.float32) # only score for now

    def __iter__(self):
        wi = get_worker_info()
        if wi is None:
            print('New iterator from the dataset')
            iterator = FileIterator(self.feat_files, self.targ_files, self.rezult_target,
                    self.evaluate, self.skip, 0, 1)
        else:
            wiid = wi.id
            winw = wi.num_workers
            print(f'New iterator from the dataset: {wiid} / {winw}')
            iterator = FileIterator(self.feat_files, self.targ_files, self.rezult_target,
                    self.evaluate, self.skip, wiid, winw)
        return iterator

class FileIterator():
    def __init__(self, feature_files, target_files, rezult_target, evaluate, skip, wid, wnum):
        self.feature_files = feature_files
        self.target_files = target_files
        self.target_idx = 1 if rezult_target else 0
        self.evaluate = evaluate
        self.skip = skip
        self.wid = wid      # worker id
        self.wnum = wnum    # number of workers
        self.feo_file = None
        self.tao_file = None
        self.cur_file = None
        self.cur_rec = 0

    def __next__(self):
        # It would work even with empty files
        # Because we have features and target, the shorter file will terminate
        # that pair and begin read from the next pair
        while True:
            new_file = False
            if self.feo_file is None:
                assert self.tao_file is None
                self.cur_file = 0
                new_file = True
            else:
                try:
                    fe_line = next(self.feo_file)
                    if not self.evaluate:
                        ta_line = next(self.tao_file)
                except StopIteration:
                    self.cur_file += 1
                    if self.cur_file >= len(self.feature_files):
                        raise StopIteration
                    new_file = True
            if new_file:
                if self.wnum > 1:
                    while self.cur_file % self.wnum != self.wid:
                        self.cur_file += 1
                        if self.cur_file >= len(self.feature_files):
                            raise StopIteration
                if self.feo_file is not None:
                    self.feo_file.close()
                print(f'W{self.wid}: open feature file {self.cur_file}: {self.feature_files[self.cur_file]}')
                self.feo_file = open(self.feature_files[self.cur_file], 'r')
                if not self.evaluate:
                    if self.tao_file is not None:
                        self.tao_file.close()
                    print(f'W{self.wid}: open target  file {self.cur_file}: {self.target_files[self.cur_file]}')
                    self.tao_file = open(self.target_files[self.cur_file], 'r')
            else:
                self.cur_rec += 1
                if self.cur_rec > self.skip:
                    break

        # We must set 1 on the feature indices
        feix = np.fromstring(fe_line, dtype=int, sep=',')
        feat = np.zeros(FEATURE_LEN, dtype=np.float32)
        feat[feix] = 1.0
        if self.evaluate:
            return torch.as_tensor(feat)
        else:
            targ = np.fromstring(ta_line, dtype=np.float32, sep=',')[self.target_idx]
            return torch.as_tensor(feat), torch.as_tensor(targ)

if __name__ == '__main__':
    print('Test dataset')
    train_dir = 'C:\\data\\extract\\2025\\beenine\\train'
    tds = BBNNDataset(train_dir)
    i = 0
    for y in tds:
        print(f'{y}')
        #print(f'Shapes: {X.shape} {X.type} {y.shape} {y.type}')
        i += 1
        if i >= 2:
            break
