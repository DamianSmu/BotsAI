import requests
import Constants


class BotController:
    def __init__(self, name):
        self.name = name
        self.session = requests.Session()
        self.token = None

    def signup(self):
        payload = {
            "username": self.name,
            "email": self.name + "_email",
            "password": self.name + "_pass"
        }
        try:
            r = requests.post(url=Constants.BASE_URL + Constants.SIGNUP_URL, json=payload)
            if r.status_code != requests.codes.unprocessable:
                return
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