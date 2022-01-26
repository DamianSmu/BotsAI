import logging
import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

import Constants
from Bot import Bot
from Game import Game
from NN import NN
from Q import Q
from Reward import attack_strategy_reward
from TestBot import TestBot

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print(tf.__version__)
print('A: ', tf.test.is_built_with_cuda)
print('B: ', tf.test.gpu_device_name())
local_device_protos = device_lib.list_local_devices()
print([x.name for x in local_device_protos if x.device_type == 'GPU'],
      [x.name for x in local_device_protos if x.device_type == 'CPU'])

dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.mkdir(dir_name)
logging.basicConfig(filename=dir_name + '/log.txt', encoding='utf-8', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

memSize = 1000
batch_size = 10
learningRate = 0.001
gamma = 0.99
epsilon = 1.
epsilonDecayRate = 0.0002
min_epsilon = 0.01
filepathToSave = dir_name + '/model_attack.h5'
mapSize = 20
actions_number = len(Constants.actions)
nn_predict = NN(map_size=mapSize, local_size=3, input_depth=7, actions=11, lr=learningRate)
model_predict = nn_predict.model

nn_target = NN(map_size=mapSize, local_size=3, input_depth=7, actions=11, lr=learningRate)
model_target = nn_target.model

maxEpochs = 5000
maxSteps = 40
scores = []
total_reward = 0
total_steps = 0
max_reward = 0
bots_count = 2
copy_weights = 1000
warmup = 5000
plot_x = []
plot_y = []
loss = []
current_loss = 0

q = Q(memSize, gamma)
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

    # Charts vars
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
            print("break")
            break

        global_state, local_state = bot.get_padded_state(bot.units[0])
        action = bot.predict_action(model_predict, bot.units[0], epsilon, global_state, local_state)
        response = game.post_actions_and_take_turn(action, bot)
        current_state_next = bot.set_state(game.get_objects(), mapSize)
        global_state_next, local_state_next = bot.get_padded_state(bot.units[0])
        invalid_actions = response['invalidActions']

        reward = attack_strategy_reward(current_state, current_state_next, local_state, action, invalid_actions)
        q.save([[global_state, local_state], action, reward, [global_state_next, local_state_next]], gameOver)

        posx.append(bot.units[0]['x'])
        posy.append(bot.units[0]['y'])
        colors.append(step)

        posx_o.append(opponent.bot.units[0]['x'])
        posy_o.append(opponent.bot.units[0]['y'])
        colors_o.append(step)

        if total_steps > warmup:
            current_loss += q.train_on_batch(model_predict, model_target, batch_size)

        log_entry = str(epoch) + ";" + str(step) + ";" + str(epsilon) + ";" + str(action) + ";" + str(
            invalid_actions) + ";" + str(reward)
        logging.info(log_entry)

        if total_steps < warmup:
            time.sleep(0.002)

        if epsilon > min_epsilon and total_steps > warmup:
            epsilon -= epsilonDecayRate

        if step >= maxSteps:
            gameOver = True

        if len(opponent.bot.units) == 0:
            gameOver = True

        if reward == 5:
            gameOver = True

        if total_steps % copy_weights == 0:
            model_target.set_weights(model_predict.get_weights())

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
            # plot_x.append(epoch)
            # plot_y.append(step)
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
