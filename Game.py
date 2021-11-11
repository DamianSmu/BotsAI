import numpy as np
import requests

import Constants
from Map import Map


class Game:
    def __init__(self, owner):
        self.id = None
        self.owner = owner
        self.session = requests.Session()
        self.map = Map()
        self.players = [owner]
        if owner.token:
            self.session.headers.update({'Authorization': 'Bearer ' + owner.token})
        else:
            raise Exception("Owner does not have auth token")

    def create(self):
        try:
            r = self.session.post(url=Constants.BASE_URL + Constants.GAME_URL)
            if r.ok:
                self.id = r.text
                return r.text
        except requests.exceptions.HTTPError as e:
            print("Request error: ", e)

    def start(self):
        try:
            r = self.session.post(url=Constants.BASE_URL + Constants.GAME_URL + self.id + Constants.GAME_START_URL)
            if r.ok:
                self.map = Map()
                self.map.set_terrain(self.getMap())
        except requests.exceptions.HTTPError as e:
            print("Request error: ", e)

    def status(self):
        try:
            r = self.session.get(url=Constants.BASE_URL + Constants.GAME_URL + self.id + Constants.GAME_STATUS_URL)
            r.raise_for_status()
            if r.ok and r.text:
                return r.text
        except requests.exceptions.HTTPError as e:
            print("Request error: ", e)

    def connect_player(self, player):
        try:
            if player.id == self.owner.id:
                return
            else:
                r = requests.post(url=Constants.BASE_URL + Constants.GAME_URL + self.id + Constants.GAME_CONNECT,
                                  headers={'Authorization': 'Bearer ' + player.token})
                if r.ok:
                    self.players.append(player)
        except requests.exceptions.HTTPError as e:
            print("Request error: ", e)

    def getMap(self):
        try:
            return self.session.get(url=Constants.BASE_URL + Constants.GAME_URL + self.id).json()
        except requests.exceptions.HTTPError as e:
            print("Request error: ", e)

    def getObjects(self):
        try:
            return self.session.get(
                url=Constants.BASE_URL + Constants.GAME_URL + self.id + Constants.GAME_MAP_OBJECTS).json()
        except requests.exceptions.HTTPError as e:
            print("Request error: ", e)


    def takeTurn(self):
        try:
            r = self.session.post(url=Constants.BASE_URL + Constants.GAME_URL + self.id + Constants.GAME_TAKE_TURN)
        except requests.exceptions.HTTPError as e:
            print("Request error: ", e)
