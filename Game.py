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
        r = requests.post(url=Constants.BASE_URL + Constants.GAME_URL,
                          headers={'Authorization': 'Bearer ' + self.owner.token})
        r.raise_for_status()
        if r.ok:
            self.id = r.text
            return r.text

    def start(self):
        if self.owner is None:
            raise Exception("Owner is not set")
        r = requests.post(url=Constants.BASE_URL + Constants.GAME_URL + self.id + Constants.GAME_START_URL,
                          headers={'Authorization': 'Bearer ' + self.owner.token})
        r.raise_for_status()
        if r.ok:
            self.map = Map()
            self.map.set_terrain(self.get_map())
            for b in self.bots:
                b.terrain = self.map.terrain

    def status(self):
        r = requests.get(url=Constants.BASE_URL + Constants.GAME_URL + self.id + Constants.GAME_STATUS_URL,
                         headers={'Authorization': 'Bearer ' + self.owner.token})
        r.raise_for_status()
        if r.ok and r.text:
            return r.text

    def connect_bot(self, bot):
        if bot.id == self.owner.id:
            return
        else:
            r = requests.post(url=Constants.BASE_URL + Constants.GAME_URL + self.id + Constants.GAME_CONNECT,
                              headers={'Authorization': 'Bearer ' + bot.token})
            r.raise_for_status()
            if r.ok:
                self.bots.append(bot)

    def get_map(self):
        r = requests.get(url=Constants.BASE_URL + Constants.GAME_URL + self.id,
                            headers={'Authorization': 'Bearer ' + self.owner.token})
        r.raise_for_status()
        return r.json()

    def get_objects(self):
        r = requests.get(
            url=Constants.BASE_URL + Constants.GAME_URL + self.id + Constants.GAME_MAP_OBJECTS,
            headers={'Authorization': 'Bearer ' + self.owner.token})
        r.raise_for_status()
        return r.json()

    def signup_bot(self, bot):
        payload = {
            "username": bot.name,
            "email": bot.name + "_email",
            "password": bot.name + "_pass"
        }
        r = requests.post(url=Constants.BASE_URL + Constants.SIGNUP_URL, json=payload)
        r.raise_for_status()
        if r.ok:
            bot.id = r.json()['id']
            bot.token = r.json()['token']

    def signin_bot(self, bot):
        payload = {
            "username": bot.name,
            "password": bot.name + "_pass"
        }
        r = requests.post(url=Constants.BASE_URL + Constants.SIGNIN_URL, json=payload)
        r.raise_for_status()
        if r.ok and r.text:
            bot.token = r.text

    def get_bot(self, bot):
        return requests.get(url=Constants.BASE_URL + Constants.ME_URL,
                            headers={'Authorization': 'Bearer ' + bot.token}).json()

    def post_action(self, action_list, bot):
        payload = []
        for a in action_list:
            j = {
                "actionType": Constants.actions[a[2]],
                "x": a['x'],
                "y": a['y'],
            }
            payload.append(j)
        r = requests.post(url=Constants.BASE_URL + Constants.GAME_URL + self.id + Constants.GAME_ACTION,
                          json=payload,
                          headers={'Authorization': 'Bearer ' + bot.token})
        r.raise_for_status()
        return r

    def take_turn(self, bot):
        r = requests.post(url=Constants.BASE_URL + Constants.GAME_URL + self.id + Constants.GAME_TAKE_TURN,
                          headers={'Authorization': 'Bearer ' + bot.token})
        r.raise_for_status()
        return r.json()

    def end_turn(self, bot):
        r = requests.post(url=Constants.BASE_URL + Constants.GAME_URL + self.id + Constants.GAME_END_TURN,
                          headers={'Authorization': 'Bearer ' + bot.token})
        r.raise_for_status()

    def post_actions_and_take_turn(self, action, bot):
        payload = []
        j = {
            "actionType": Constants.actions[action[2]],
            "x": action[0],
            "y": action[1],
        }
        payload.append(j)

        r = requests.post(
            url=Constants.BASE_URL + Constants.GAME_URL + self.id + Constants.GAME_POST_ACTION_AND_TAKE_TURN,
            json=payload,
            headers={'Authorization': 'Bearer ' + bot.token})
        r.raise_for_status()
        return r.json()

    def set_owner(self, bot):
        if bot.token is None:
            raise Exception("Bot has no token")
        self.owner = bot
        self.bots.append(bot)
