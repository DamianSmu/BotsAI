import os
from datetime import datetime
import time
import Constants
from Bot import Bot
from Game import Game
from NN import NN
from Q import Q
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.client import device_lib


def find_closest(state, state_next):
    res_list = np.argwhere((state[:, :, 0]) == 10 & (state[:, :, 6] != 10))
    res_list_next = np.argwhere((state_next[:, :, 0]) == 10 & (state_next[:, :, 6] != 10))
    x, y = tuple(np.argwhere(state[:, :, 1] == 1)[0])
    xn, yn = tuple(np.argwhere(state_next[:, :, 1] == 1)[0])

    closest = 40
    for r in res_list:
        xr, yr = r
        if abs(x - xr) + abs(y - yr) < closest:
            closest = abs(x - xr) + abs(y - yr)

    closest_next = 40
    for r in res_list_next:
        xr, yr = r
        if abs(xn - xr) + abs(yn - yr) < closest_next:
            closest_next = abs(xn - xr) + abs(yn - yr)
    return (closest - closest_next) / 4

from TestBot import TestBot
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
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

memSize = 5000
batch_size = 16
learningRate = 0.001
gamma = 0.98
epsilon = 1.
epsilonDecayRate = 0.0002
min_epsilon = 0.05
filepathToSave = 'model.h5'
mapSize = 20
actions_number = len(Constants.actions)
input_state_layers = 11
nn_predict = NN(map_size=mapSize, local_size=3, depth=7, actions_number=11, lr=learningRate)
model_predict = nn_predict.model

nn_target = NN(map_size=mapSize, local_size=3, depth=7, actions_number=11, lr=learningRate)
model_target = nn_target.model

maxEpochs = 5000
maxSteps = 40
scores = []
total_reward = 0
total_steps = 0
max_reward = 0
bots_count = 1
copy_weights = 1000
warmup = 5000
plot_x = []
plot_y = []
log = []
loss = []
current_loss = 0

q = Q(memSize, gamma)
game = Game()
bots = []
for i in range(bots_count):
    bots.append(Bot('player_' + str(i)))
    game.signup_bot(bots[i])

game.set_owner(bots[0])
bot = bots[0]
# opponent = TestBot(game, None, bots[1], True)
game.create()
for i in range(1, bots_count):
    game.connect_bot(bots[i])

epoch = 0
for epoch in range(maxEpochs):
    gameOver = False
    step = 0
    placed = 0
    game.start()

    # scattter
    posx = []
    posy = []
    colors = []
    sA = np.where(game.map.terrain < 4, 0, game.map.terrain).T

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
        invalidActions = response['invalidActions']
        reward = 0
        reward_for_distance = 0
        if invalidActions == 0:
            reward_for_distance = find_closest(state=current_state, state_next=current_state_next)
            # Calculate reward for placed settlement
            if action[2] == 8:
                reward = 0.5
                if local_state[1, 1, 0] > 3:
                    reward = 5
                    placed = step
            else:
                reward = reward_for_distance
        else:
            reward = -1

        q.remember([[global_state, local_state], action, reward, [global_state_next, local_state_next]], gameOver)

        if total_steps > warmup:
            current_loss += q.train_on_batch(model_predict, model_target, batch_size)

        log_entry = ""
        log_entry += "epoch: " + str(epoch).ljust(5)
        log_entry += ";step: " + str(step).ljust(3)
        log_entry += ";eps: " + str("%.4f" % epsilon).ljust(8)
        log_entry += ";action: " + str(action).ljust(14)
        log_entry += ";invalidActions: " + str(invalidActions).ljust(2)
        log_entry += ";reward: " + str(reward) + '\n'
        log.append(log_entry)
        print(log_entry)

        if total_steps < warmup:
            time.sleep(0.002)

        if epsilon > min_epsilon and total_steps > warmup:
            epsilon -= epsilonDecayRate

        if step >= maxSteps:
            gameOver = True

        if total_steps % copy_weights == 0:
            model_target.set_weights(model_predict.get_weights())

        total_reward += reward
        # if reward > max_reward:
        #     max_reward = reward
        #     model_predict.save(filepathToSave)

        game.end_turn(bot)
        if gameOver:
            plt.imshow(sA, interpolation='none', aspect='equal', cmap='Blues')
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

            # Save log to file
            output_file = open(dir_name + '\\log.txt', 'w')
            for entry in log:
                output_file.write(entry)
            output_file.close()

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
            model_predict.save(filepathToSave)


