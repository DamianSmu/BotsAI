import numpy as np

terrain = {'WATER': 1,
           'LAND': 3,
           'IRON': 5,
           'GOLD': 10}


class Map:
    def __init__(self):
        self.terrain = None
        self.size = None

    def set_terrain(self, json):
        j = json['map']['tiles']
        size = json['map']['size']
        self.size = size
        arr = np.empty((size, size))
        for x in j:
            arr[int(x['position']['x'])][int(x['position']['y'])] = terrain[x['type']]
        self.terrain = arr
