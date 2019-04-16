import pickle
import torch

import collections
import itertools
import math
import os
import random

import torch

from networks import ConvEncoder, ConvDecoder
from losses import LabelSmoothedCrossEntropyCriterion

import models

from fairseq import Dictionary
def make_model():
    source_dictionary = Dictionary.load("data/data-bin/writingPrompts/dict.wp_source.txt")
    target_dictionary = Dictionary.load("data/data-bin/writingPrompts/dict.wp_target.txt")

    encoder = ConvEncoder(
        source_dictionary,
        embed_dim=256,
        convolutions=[(128, 3)] * 2 + [(512,3)] * 1,
        dropout=0.1,
        max_positions=1500,
        attention=True,
        attention_nheads=1
    )

    decoder = ConvDecoder(
        target_dictionary,
        embed_dim=256,
        convolutions=[(512, 4)] * 4 + [(768, 4)] * 2 + [(1024, 4)] * 1,
        out_embed_dim=256,
        attention=True,
        dropout=0.1,
        max_positions=1500,
        selfattention=True,
        attention_nheads=1,
        selfattention_nheads=4,
        project_input=True,
        gated_attention=True,
        downsample=True,
    )

    model = models.BaseModel(encoder, decoder)
    return model


def main():
    model = make_model()
    loss = LabelSmoothedCrossEntropyCriterion



