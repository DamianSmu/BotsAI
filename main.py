# %%
from BotController import BotController
from GameController import GameController
from Map import Map

bot1 = BotController('bot1')
bot1.signup()
bot1.signin()
print(bot1.get())

game = GameController(bot1)
game.create()
print(game.status())
game.start()
print(game.status())
# %%
v = game.getMap();
map = Map()

print(map.parse(v))

