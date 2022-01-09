import logging
import os
import sys
from datetime import datetime
import time
import Constants
from Bot import Bot
from Game import Game
from NN import NN
from Q import Q
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.client import device_lib

from Reward import settlements_strategy_reward
from TestBot import TestBot
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print(tf.__version__)
print('A: ', tf.test.is_built_with_cuda)
print('B: ', tf.test.gpu_device_name())
local_device_protos = device_lib.list_local_devices()
print([x.name for x in local_device_protos if x.device_type == 'GPU'],
      [x.name for x in local_device_protos if x.device_type == 'CPU'])

# physical_devices = tf.config.list_physical_devices('GPU')
# for device in physical_devices:
#     tf.config.experimental.set_memory_growth(device, True)

dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.mkdir(dir_name)
logging.basicConfig(filename=dir_name + '/log.txt', encoding='utf-8', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

memSize = 10000
batch_size = 10
learningRate = 0.001
gamma = 0.99
epsilon = 1.
epsilonDecayRate = 0.0002
min_epsilon = 0.01
model_path = dir_name + '/model_set.h5'
map_size = 20
actions_number = len(Constants.actions)
nn_predict = NN(map_size=map_size, local_size=3, input_depth=7, actions_number=11, lr=learningRate)
model_predict = nn_predict.model

nn_target = NN(map_size=map_size, local_size=3, input_depth=7, actions_number=11, lr=learningRate)
model_target = nn_target.model

maxEpochs = 1000
max_steps = 40
scores = []
total_reward = 0
total_steps = 0
copy_weights = 1000
warmup = 5000
plot_x = []
plot_y = []
loss = []
current_loss = 0

q = Q(memSize, gamma)
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
        current_state = bot.set_state(game.get_objects(), map_size)

        posx.append(bot.units[0]['x'])
        posy.append(bot.units[0]['y'])
        colors.append(step)

        global_state, local_state = bot.get_padded_state(bot.units[0])
        action = bot.predict_action(model_predict, bot.units[0], epsilon, global_state, local_state)
        response = game.post_actions_and_take_turn(action, bot)
        current_state_next = bot.set_state(game.get_objects(), map_size)
        global_state_next, local_state_next = bot.get_padded_state(bot.units[0])
        invalid_actions = response['invalidActions']

        reward = settlements_strategy_reward(current_state, current_state_next, local_state, action, invalid_actions)

        q.remember([[global_state, local_state], action, reward, [global_state_next, local_state_next]], gameOver)

        if total_steps > warmup:
            current_loss += q.train_on_batch(model_predict, model_target, batch_size)

        log_entry = str(epoch) + ";" + str(step) + ";" + str(epsilon) + ";" + str(action) + ";" + str(invalid_actions) + ";" + str(reward)
        logging.info(log_entry)

        if total_steps < warmup:
            time.sleep(0.002)

        if epsilon > min_epsilon and total_steps > warmup:
            epsilon -= epsilonDecayRate

        if step >= max_steps:
            gameOver = True

        if total_steps % copy_weights == 0:
            model_target.set_weights(model_predict.get_weights())

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

        if total_steps % max_steps == 0:
            scores.append(total_reward)
            total_reward = 0
            plt.plot(np.convolve(scores, np.ones(5)/5, mode='valid'))
            plt.grid()
            plt.savefig(dir_name + '\\reward.png')
            plt.close()
            model_predict.save(model_path)

        if total_steps % 10 == 0:
            loss.append(current_loss / 10)
            current_loss = 0
            plt.plot(loss)
            plt.grid()
            plt.savefig(dir_name + '\\loss.png')
            plt.close()