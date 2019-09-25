import sys
import os

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))

import json
from os.path import dirname, abspath
from helpers.vqa import *


class VQALoader:
    def __init__(self, version='train'):

        with open(os.path.join(this_path, 'structure.json')) as fp:
            structure = json.load(fp)

        self.dir = dirname(dirname(abspath(__file__)))

        # Loading strings
        self.v = structure['version']
        self.t = structure['task']
        self.o = structure['objective'][version]
        self.d = structure['data']
        self.p = {
            'q': structure['path']['questions'],
            'a': structure['path']['answers'],
            'i': structure['path']['images']
        }

        self.q = self.p['q'].format(self.dir, self.v, self.t,  self.d, self.o)
        self.a = self.p['a'].format(self.dir, self.v, self.d, self.o)
        self.i = self.p['i'].format(self.dir, self.v, self.o)

    def load(self):
        return VQA(self.a, self.q)


if __name__ == '__main__':
    VQALoader(version='test').load()
