# %%

from Game import Game
from Player import Player

players = []
for i in range(10):
    players.append(Player('player_' + str(i)))
    players[i].signup()
    players[i].signin()

game = Game(players[0])
game.create()
for i in range(1, 10):
    game.connect_player(players[i])

game.start()
print(game.status())
# %%
objects = game.getObjects()

for p in players:
    p.updateObjects(objects, game.map.size)
    for k, b in p.bots.items():
        # b.calculate_reward()
        print(str(p.name) + ": " + str(p.calculate_reward()))

# %%
players[0].postAction(game.id, 8, next(iter(players[0].bots.values())))
game.takeTurn()

objects = game.getObjects()
for p in players:
    p.updateObjects(objects, game.map.size)
    for k, b in p.bots.items():
        print(str(p.name) + ": " + str(p.calculate_reward()) + ": " + str(b.bot_matrix))
