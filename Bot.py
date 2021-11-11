import numpy as np


class Bot:
    def __init__(self, position, defence, bot_matrix):
        self.objects = None
        self.position = position
        self.defence = defence
        self.type = None
        self.reward = 0
        self.bot_matrix = bot_matrix

