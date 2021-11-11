import numpy as np
import requests

import Constants
from Bot import Bot


class Player:
    def __init__(self, name):
        self.name = name
        self.bots = {}
        self.id = None
        self.session = requests.Session()
        self.token = None
        self.player_matrix = None

    def signup(self):
        payload = {
            "username": self.name,
            "email": self.name + "_email",
            "password": self.name + "_pass"
        }
        try:
            r = requests.post(url=Constants.BASE_URL + Constants.SIGNUP_URL, json=payload)
            if r.ok:
                self.id = r.json()['id']
        except requests.exceptions.HTTPError as e:
            print("Request error: ", e)

    def signin(self):
        payload = {
            "username": self.name,
            "password": self.name + "_pass"
        }
        try:
            r = requests.post(url=Constants.BASE_URL + Constants.SIGNIN_URL, json=payload)
            if r.ok and r.text:
                self.session.headers.update({'Authorization': 'Bearer ' + r.text})
                self.token = r.text
        except requests.exceptions.HTTPError as e:
            print("Request error: ", e)

    def get(self):
        try:
            return self.session.get(url=Constants.BASE_URL + Constants.ME_URL).json()
        except requests.exceptions.HTTPError as e:
            print("Request error: ", e)

    def postAction(self, game_id, action, bot):
        payload = {
            "actionType": Constants.actions[action],
            "from": {
                "x": bot.position[0],
                "y": bot.position[1]
            }
        }
        try:
            r = self.session.post(url=Constants.BASE_URL + Constants.GAME_URL + game_id + Constants.GAME_ACTION,
                                  json=payload)
        except requests.exceptions.HTTPError as e:
            print("Request error: ", e)

    def updateObjects(self, objects, map_size):
        arr = np.zeros((map_size, map_size))
        for obj in objects:
            map_object = obj['mapObject']
            if map_object is not None:
                if map_object['playerSession']['user']['id'] == self.id:
                    arr[int(obj['position']['x'])][int(obj['position']['y'])] = map_object['defence']
                    position = (int(obj['position']['x']), int(obj['position']['y']))
                    self.bots = {}
                    bot_matrix = np.zeros((map_size, map_size))
                    bot_matrix[position] = map_object['defence']
                    self.bots.update({map_object['id']: Bot(position, map_object['defence'], bot_matrix)})
                else:
                    arr[int(obj['position']['x'])][int(obj['position']['y'])] = -map_object['defence']
        self.player_matrix = arr

    def calculate_reward(self):
        return self.player_matrix.sum()
