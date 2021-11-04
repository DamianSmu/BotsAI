import json
import sys

import numpy as np

types = { 'WATER': 0,
          'LAND': 1,
          'IRON': 2,
          'GOLD': 3}


class Map:
    def __init__(self):
        self.terrain = None

    def parse(self, json):
        j = json['map']['tiles']
        size = json['map']['size']
        arr = np.empty((size, size))
        for x in j:
            arr[int(x['position']['x'])][int(x['position']['y'])] = types[x['type']]
        self.terrain = arr

    def update(self):
        pass