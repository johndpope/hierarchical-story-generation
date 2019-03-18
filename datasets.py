import numpy as np
import torch
import pickle
import csv
import os
from torch.utils.data import Dataset


class WritingPrompts(Dataset):
    def __init__(self, root='./data/writingPrompts', train=True):
        self.train = train

        if not os.path.isdir("./data/pickles"):
            os.mkdir(os.fsencode("./data/pickles"))

        source_pickle = './data/pickles/' + ('train' if self.train else 'val') + '_tokenized_source.p'
        target_pickle = './data/pickles/' + ('train' if self.train else 'val') + '_tokenized_target.p'
        cached = os.path.isfile(source_pickle)aa

        if not cached:
            if self.train:
                source_file = root + "/train.wp_source"
                target_file = root + "/train.wp_target"
            else:
                source_file = root + "/valid.wp_source"
                target_file = root + "/valid.wp_source"

            with open(source_file) as f:
                self.source_data = [s.split(' ') for s in f.readlines()]

            with open(target_file) as f:
                self.target_data = [s.split(' ') for s in f.readlines()]

            pickle.dump(self.source_data, open(source_pickle, 'wb'))
            pickle.dump(self.target_data, open(target_pickle, 'wb'))
        else:
            self.source_data = pickle.load(open(source_pickle, 'rb'))
            self.target_data = pickle.load(open(target_pickle, 'rb'))

