# %%
import numpy as np

from Bot import Bot
from Game import Game
from Map import Map

bot1 = Bot('bot1')
bot1.signup()
bot1.signin()



bot2 = Bot('bot2')
bot2.signup()
bot2.signin()

game = Game(bot1)
game.create()
game.connect_bot(bot2)
print(game.status())
game.start()
print(game.status())
# %%
game.updateObjects()
print(bot1.objects[np.where(bot1.objects == -10)])
