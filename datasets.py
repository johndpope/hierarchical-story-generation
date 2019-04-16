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

    ######################################

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, src, tgt, lang, data_path):
            filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedDataset.exists(filename):
                return True
            return False

        def indexed_dataset(path, dictionary):
            if self.args.raw_text:
                return IndexedRawTextDataset(path, dictionary)
            elif IndexedDataset.exists(path):
                if self.args.lazy_load:
                    return IndexedDataset(path, fix_lua_indexing=True)
                else:
                    return IndexedCachedDataset(path, fix_lua_indexing=True)
            return None

        src_datasets = []
        tgt_datasets = []

        data_paths = self.args.data

        for dk, data_path in enumerate(data_paths):
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')

                # infer langcode
                src, tgt = self.args.source_lang, self.args.target_lang
                if split_exists(split_k, src, tgt, src, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
                elif split_exists(split_k, tgt, src, src, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
                else:
                    if k > 0 or dk > 0:
                        break
                    else:
                        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

                src_datasets.append(indexed_dataset(prefix + src, self.src_dict))
                tgt_datasets.append(indexed_dataset(prefix + tgt, self.tgt_dict))

                print('| {} {} {} examples'.format(data_path, split_k, len(src_datasets[-1])))

                if not combine:
                    break

        assert len(src_datasets) == len(tgt_datasets)

        if len(src_datasets) == 1:
            src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
        else:
            sample_ratios = [1] * len(src_datasets)
            sample_ratios[0] = self.args.upsample_primary
            src_dataset = ConcatDataset(src_datasets, sample_ratios)
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

        self.datasets[split] = LanguagePairDataset(
            src_dataset, src_dataset.sizes, self.src_dict,
            tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
        )

