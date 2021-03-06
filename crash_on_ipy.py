# -*- coding: utf-8 -*-
import sys


# noinspection PyClassHasNoInit
class ExceptionHook:
    instance = None

    def __call__(self, *args, **kwargs):
        if self.instance is None:
            from IPython.core import ultratb
            self.instance = ultratb.FormattedTB(
                mode='Plain', color_scheme='Linux', call_pdb=1)
        # noinspection PyCallingNonCallable
        return self.instance(*args, **kwargs)


def init():
    sys.excepthook = ExceptionHook()
