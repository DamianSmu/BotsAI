import logging
import os
import sys
from datetime import datetime
import time

from keras.models import load_model

import Constants
from Bot import Bot
from Game import Game
from NN import NN
from Q import Q
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.client import device_lib
from Reward import attack_strategy_reward
from TestBot import TestBot


dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.mkdir(dir_name)
logging.basicConfig(filename=dir_name + '/log.txt', encoding='utf-8', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

epsilon = 0
filepathToSave = dir_name + '/model_attack.h5'
mapSize = 20
actions_number = len(Constants.actions)

maxEpochs = 100
maxSteps = 40
scores = []
total_reward = 0
total_steps = 0
bots_count = 2
plot_x = []
plot_y = []
loss = []
current_loss = 0
# Path to model
model_predict = load_model(filepath="C:\\Users\\Damian\\PycharmProjects\\BotsAI\\2022-01-04_16-54-37\\model_attack.h5")

game = Game()
bots = []
bots.append(Bot('player_0'))
bots.append(Bot('player_1'))
game.signup_bot(bots[0])
game.signup_bot(bots[1])

game.set_owner(bots[0])
bot = bots[1]
opponent = TestBot(game, None, bots[0], True, None, [8])
game.create()
game.connect_bot(bots[1])

for epoch in range(maxEpochs):
    gameOver = False
    step = 0
    game.start()

    posx = []
    posy = []
    colors = []
    posx_o = []
    posy_o = []
    colors_o = []

    while not gameOver:
        opponent.play()

        step += 1
        total_steps += 1
        current_state = bot.set_state(game.get_objects(), mapSize)
        if len(opponent.bot.units) == 0 or len(bot.units) == 0:
            break

        global_state, local_state = bot.get_padded_state(bot.units[0])
        action = bot.predict_action(model_predict, bot.units[0], epsilon, global_state, local_state)
        response = game.post_actions_and_take_turn(action, bot)
        current_state_next = bot.set_state(game.get_objects(), mapSize)
        global_state_next, local_state_next = bot.get_padded_state(bot.units[0])
        invalid_actions = response['invalidActions']

        reward = attack_strategy_reward(current_state, current_state_next, local_state, action, invalid_actions)

        posx.append(bot.units[0]['x'])
        posy.append(bot.units[0]['y'])
        colors.append(step)

        posx_o.append(opponent.bot.units[0]['x'])
        posy_o.append(opponent.bot.units[0]['y'])
        colors_o.append(step)

        log_entry = str(epoch) + ";" + str(step) + ";" + str(epsilon) + ";" + str(action) + ";" + str(
            invalid_actions) + ";" + str(reward)
        logging.info(log_entry)

        if step >= maxSteps:
            gameOver = True

        if len(opponent.bot.units) == 0:
            gameOver = True

        if reward == 5:
            gameOver = True

        total_reward += reward

        game.end_turn(bot)
        if gameOver:
            plt.plot(posx, posy)
            plt.scatter(posx, posy, c=colors, cmap='copper_r')
            plt.plot(posx_o, posy_o)
            plt.scatter(posx_o, posy_o, c=colors_o, cmap='Reds')
            if action[2] in [4, 5, 6, 7] and reward == 5:
                plt.plot([posx[-1], posx_o[-1]], [posy[-1], posy_o[-1]], 'mD-', )

            ax = plt.gca()
            ax.grid(color='black', alpha=0.5, linestyle='-', linewidth=0.5)
            ax.set_xticks(np.arange(0, 20, 1))
            ax.set_yticks(np.arange(0, 20, 1))
            ax.set_xticklabels(np.arange(0, 20, 1), fontsize=6)
            ax.set_yticklabels(np.arange(0, 20, 1), fontsize=6)
            plt.rc('axes', labelsize=20)
            plt.savefig(dir_name + '\\stats_pos_' + str(epoch) + '.png', dpi=200)
            plt.close()

        if total_steps % 80 == 0:
            scores.append(total_reward / 2)
            total_reward = 0
            plt.plot(scores)
            plt.savefig(dir_name + '\\reward.png')
            plt.close()
            model_predict.save(filepathToSave)

        if total_steps % 10 == 0:
            loss.append(current_loss / 10)
            current_loss = 0
            plt.plot(loss)
            plt.savefig(dir_name + '\\loss.png')
            plt.close()
