# vim: ts=4 sw=4 et

import os.path
import numpy as np
import glob

FEATURE_LEN = 384 * 2

class CountFeatures():
    def __init__(self, data_dir):
        self.feat = np.zeros(FEATURE_LEN, dtype=int)
        self.feat_files = self.filenames(data_dir, '*-feat.txt')
        assert len(self.feat_files) != 0

    def filenames(self, data_dir, pat):
        found = []
        #print(f'filenames in {data_dir} with {pat}')
        for file_name in glob.glob(pat, root_dir=data_dir):
            abs_name = os.path.join(data_dir, file_name)
            found.append(abs_name)
        return found

    def count(self):
        for fn in self.feat_files:
            print(f'{fn}', end='')
            with open(fn, 'r') as f:
                i = 0
                for line in f:
                    # We add 1 on the feature indices
                    feix = np.fromstring(line, dtype=int, sep=',')
                    self.feat[feix] += 1
                    i += 1
            print(f' -> {i} lines')

if __name__ == '__main__':
    train_dir = r'E:\extract\2025\test-1\train'
    print(f'Count dataset {train_dir}')
    cds = CountFeatures(train_dir)
    cds.count()
    print('Feature counts:')
    print(cds.feat)
