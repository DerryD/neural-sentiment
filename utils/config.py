# -*- coding: utf-8 -*-
class Config(object):
    def __init__(self):
        self.learning_rate = 0.009
        self.max_grad_norm = 5
        self.num_layers = 2        # number of stacked LSTM cells
        self.embedding_dims = 200  # embedded size
        self.max_epoch = 50        # Number of epochs for iteration
        self.keep_prob = 0.5
        self.lr_decay = 0.97
        self.batch_size = 50
        self.num_classes = 2
        self.vocab_size = 20000
        self.max_len = 200
