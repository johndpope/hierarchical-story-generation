import numpy as np
import torch
import pickle
import csv
import os
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset


class WritingPrompts(Dataset):
    def __init__(self, root='./data/writingPrompts', train=True):
        self.train = train

        if not os.path.isdir("./data/pickles"):
            os.mkdir(os.fsencode("./data/pickles"))

        source_pickle = './data/pickles/' + ('train' if self.train else 'val') + '_tokenized_source.p'
        target_pickle = './data/pickles/' + ('train' if self.train else 'val') + '_tokenized_target.p'
        cached = os.path.isfile(source_pickle)

        if not cached:
            source_file = root + '/' + ('train' if self.train else 'valid') + '.wp_source'
            target_file = root + '/' + ('train' if self.train else 'valid') + '.wp_target'
            self.source_data = self._tokenize(source_file, source_pickle)
            self.target_data = self._tokenize(target_file, target_pickle)
        else:
            self.source_data = pickle.load(open(source_pickle, 'rb'))
            self.target_data = pickle.load(open(target_pickle, 'rb'))

    def _tokenize(self, file, pickle_path):
        with open(file) as f:
            data = [word_tokenize(line) for line in f.readlines()]
            pickle.dump(data, open(pickle_path, 'wb'))
        return data


