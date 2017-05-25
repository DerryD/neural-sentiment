# -*- coding: utf-8 -*-
class Config(object):
    def __init__(self, vocab, max_seq_len):
        self.hidden_size = 50
        self.epoch_fraction = .1
        self.max_epoch = 100
        self.dropout = 0.9
        self.lr = .0001
        self.layers = 2
        self.label_size = 2
        self.should_cell_dropout = True
        self.should_input_dropout = True
        self.embed_size = 50
        self.vocab = vocab
        self.batch_size = 1
        self.max_seq_len = max_seq_len
        self.anneal_threshold = 0.99
        self.anneal_by = 1.5
        self.l2 = 0.02
