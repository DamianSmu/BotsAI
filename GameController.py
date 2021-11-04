import requests
import Constants


class GameController:
    def __init__(self, owner):
        self.id = None
        self.owner = owner
        self.session = requests.Session()
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
            self.session.post(url=Constants.BASE_URL + Constants.GAME_URL + self.id + Constants.GAME_START_URL)
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

    def connect_player(self, bot):
        try:
            if bot is None:
                self.session.post(url=Constants.BASE_URL + Constants.GAME_URL + self.id)
            else:
                requests.post(url=Constants.BASE_URL + Constants.GAME_URL + self.id,
                              headers={'Authorization': 'Bearer ' + bot.token})
        except requests.exceptions.HTTPError as e:
            print("Request error: ", e)

    def getMap(self):
        try:
            return self.session.get(url=Constants.BASE_URL + Constants.GAME_URL + self.id).json()
        except requests.exceptions.HTTPError as e:
            print("Request error: ", e)

    def postAction(self, action):
        pass

    def takeTurn(self):
        pass