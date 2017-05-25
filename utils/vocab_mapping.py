# -*- coding: utf-8 -*-
# noinspection PyPep8Naming
import six.moves.cPickle as pickle


class VocabMapping(object):
    def __init__(self):
        with open("data/vocab.txt", "rb") as handle:
            self.dic = pickle.loads(handle.read())

    def get_index(self, token):
        if token in self.dic:
            return self.dic[token]
        else:
            return self.dic["<UNK>"]

    def get_size(self):
        return len(self.dic)
