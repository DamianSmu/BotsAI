import requests

import Constants
from Map import Map


class Game:
    def __init__(self):
        self.id = None
        self.owner = None
        self.map = None
        self.bots = []

    def create(self):
        if self.owner is None:
            raise Exception("Owner is not set")
        try:
            r = requests.post(url=Constants.BASE_URL + Constants.GAME_URL,
                              headers={'Authorization': 'Bearer ' + self.owner.token})
            if r.ok:
                self.id = r.text
                return r.text
        except requests.exceptions.HTTPError as e:
            print("Request error: ", e)

    def start(self):
        if self.owner is None:
            raise Exception("Owner is not set")
        try:
            r = requests.post(url=Constants.BASE_URL + Constants.GAME_URL + self.id + Constants.GAME_START_URL,
                              headers={'Authorization': 'Bearer ' + self.owner.token})
            if r.ok:
                self.map = Map()
                self.map.set_terrain(self.get_map())
                for b in self.bots:
                    b.terrain = self.map.terrain
        except requests.exceptions.HTTPError as e:
            print("Request error: ", e)

    def status(self):
        try:
            r = requests.get(url=Constants.BASE_URL + Constants.GAME_URL + self.id + Constants.GAME_STATUS_URL,
                             headers={'Authorization': 'Bearer ' + self.owner.token})
            r.raise_for_status()
            if r.ok and r.text:
                return r.text
        except requests.exceptions.HTTPError as e:
            print("Request error: ", e)

    def connect_bot(self, bot):
        try:
            if bot.id == self.owner.id:
                return
            else:
                r = requests.post(url=Constants.BASE_URL + Constants.GAME_URL + self.id + Constants.GAME_CONNECT,
                                  headers={'Authorization': 'Bearer ' + bot.token})
                if r.ok:
                    self.bots.append(bot)
        except requests.exceptions.HTTPError as e:
            print("Request error: ", e)

    def get_map(self):
        try:
            return requests.get(url=Constants.BASE_URL + Constants.GAME_URL + self.id,
                                headers={'Authorization': 'Bearer ' + self.owner.token}).json()
        except requests.exceptions.HTTPError as e:
            print("Request error: ", e)

    def get_objects(self):
        try:
            return requests.get(
                url=Constants.BASE_URL + Constants.GAME_URL + self.id + Constants.GAME_MAP_OBJECTS,
                headers={'Authorization': 'Bearer ' + self.owner.token}).json()
        except requests.exceptions.HTTPError as e:
            print("Request error: ", e)

    def signup_bot(self, bot):
        payload = {
            "username": bot.name,
            "email": bot.name + "_email",
            "password": bot.name + "_pass"
        }
        try:
            r = requests.post(url=Constants.BASE_URL + Constants.SIGNUP_URL, json=payload)
            if r.ok:
                bot.id = r.json()['id']
                bot.token = r.json()['token']
        except requests.exceptions.HTTPError as e:
            print("Request error: ", e)

    def signin_bot(self, bot):
        payload = {
            "username": bot.name,
            "password": bot.name + "_pass"
        }
        try:
            r = requests.post(url=Constants.BASE_URL + Constants.SIGNIN_URL, json=payload)
            if r.ok and r.text:
                bot.token = r.text
        except requests.exceptions.HTTPError as e:
            print("Request error: ", e)

    def get_bot(self, bot):
        try:
            return requests.get(url=Constants.BASE_URL + Constants.ME_URL,
                                headers={'Authorization': 'Bearer ' + bot.token}).json()
        except requests.exceptions.HTTPError as e:
            print("Request error: ", e)

    def post_action(self, action_list, bot):
        payload = []
        for a in action_list:
            j = {
                "actionType": Constants.actions[a['action']],
                "x": a['x'],
                "y": a['y'],
            }
            payload.append(j)
        try:
            r = requests.post(url=Constants.BASE_URL + Constants.GAME_URL + self.id + Constants.GAME_ACTION,
                              json=payload,
                              headers={'Authorization': 'Bearer ' + bot.token})
            if r.ok:
                return r
        except requests.exceptions.HTTPError as e:
            print("Request error: ", e)

    def take_turn(self, bot):
        try:
            r = requests.post(url=Constants.BASE_URL + Constants.GAME_URL + self.id + Constants.GAME_TAKE_TURN,
                              headers={'Authorization': 'Bearer ' + bot.token})
            if r.ok:
                return r.json()
        except requests.exceptions.HTTPError as e:
            print("Request error: ", e)

    def post_actions_and_take_turn(self, action_list, bot):
        payload = []
        for a in action_list:
            if a[2] != 11 and a[2] != 8:  # wait action todo
                j = {
                    "actionType": Constants.actions[a[2]],
                    "x": a[0],
                    "y": a[1],
                }
                payload.append(j)
        try:
            r = requests.post(
                url=Constants.BASE_URL + Constants.GAME_URL + self.id + Constants.GAME_POST_ACTION_AND_TAKE_TURN,
                json=payload,
                headers={'Authorization': 'Bearer ' + bot.token})
            if r.ok:
                return r.json()
        except requests.exceptions.HTTPError as e:
            print("Request error: ", e)

    def set_owner(self, bot):
        if bot.token is None:
            raise Exception("Bot has no token")
        self.owner = bot
        self.bots.append(bot)
