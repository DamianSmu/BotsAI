import os
import sys
from datetime import datetime
from keras.models import load_model
from Bot import Bot
from Game import Game
import numpy as np
import matplotlib.pyplot as plt
from Reward import settlements_strategy_reward
import logging

dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "-test"
os.mkdir(dir_name)
logging.basicConfig(filename=dir_name + '/log.txt', encoding='utf-8', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

epsilon = 0
mapSize = 20
# Path to model
model_predict = load_model(filepath="C:\\Users\\Damian\\PycharmProjects\\BotsAI\\2022-01-03_09-41-09\\model_set.h5")

maxEpochs = 100
maxSteps = 40
scores = []
total_reward = 0
total_steps = 0
max_reward = 0
plot_x = []
plot_y = []

game = Game()

bot = Bot('player_0')
game.signup_bot(bot)
game.set_owner(bot)
game.create()
game.connect_bot(bot)

for epoch in range(maxEpochs):
    gameOver = False
    step = 0
    placed = 0
    game.start()

    posx = []
    posy = []
    colors = []
    resources_map = np.where(game.map.terrain < 4, 0, game.map.terrain).T

    while not gameOver:
        step += 1
        total_steps += 1
        current_state = bot.set_state(game.get_objects(), mapSize)

        posx.append(bot.units[0]['x'])
        posy.append(bot.units[0]['y'])
        colors.append(step)

        global_state, local_state = bot.get_padded_state(bot.units[0])
        action = bot.predict_action(model_predict, bot.units[0], epsilon, global_state, local_state)
        response = game.post_actions_and_take_turn(action, bot)
        current_state_next = bot.set_state(game.get_objects(), mapSize)
        global_state_next, local_state_next = bot.get_padded_state(bot.units[0])
        invalid_actions = response['invalidActions']

        reward = settlements_strategy_reward(current_state, current_state_next, local_state, action, invalid_actions)

        log_entry = str(epoch) + ";" + str(step) + ";" + str(epsilon) + ";" + str(action) + ";" + str(invalid_actions) + ";" + str(reward)
        logging.info(log_entry)

        if step >= maxSteps:
            gameOver = True

        total_reward += reward
        game.end_turn(bot)

        if gameOver:
            settlements_map = np.where(bot.state[:, :, 5] > 0, 10, 0)[19:-19, 19:-19].T
            plt.imshow(resources_map, interpolation='none', aspect='equal', cmap='Blues')
            plt.imshow(settlements_map, interpolation='none', aspect='equal', cmap='Reds', alpha=0.5)
            plt.plot(posx, posy)
            plt.scatter(posx, posy, c=colors, cmap='copper_r')
            ax = plt.gca()
            ax.grid(color='black', alpha=0.5, linestyle='-', linewidth=0.5)
            ax.set_xticks(np.arange(0, 20, 1))
            ax.set_yticks(np.arange(0, 20, 1))
            ax.set_xticklabels(np.arange(0, 20, 1), fontsize=6)
            ax.set_yticklabels(np.arange(0, 20, 1), fontsize=6)
            plt.rc('axes', labelsize=20)
            plt.savefig(dir_name + '\\stats_pos_' + str(epoch) + '.png', dpi=200)
            plt.close()

        if total_steps % maxSteps == 0:
            scores.append(total_reward)
            total_reward = 0
            plt.plot(np.convolve(scores, np.ones(5)/5, mode='valid'))
            plt.grid()
            plt.savefig(dir_name + '\\reward.png')
            plt.close()
