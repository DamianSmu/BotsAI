# %%
import numpy as np
# %%

from Game import Game
from Bot import Bot

game = Game()

bots = []
rewards = []
bots_count = 1
for i in range(bots_count):
    bots.append(Bot('player_' + str(i)))
    game.signup_bot(bots[i])

game.set_owner(bots[0])
game.create()
for i in range(1, bots_count):
    game.connect_bot(bots[i])

game.start()
print(game.status())
# %%

# b.calculate_reward()
# print(str(p.name) + ": " + str(p.calculate_reward()))

# %%
# print(str(p.name) + ": " + str(p.calculate_reward()) + ": " + str(b.bot_matrix))

