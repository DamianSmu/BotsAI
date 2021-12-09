import os
import Constants
from Bot import Bot
from Game import Game
from NN import NN
from Q import Q
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.client import device_lib

from TestBot import TestBot

print(tf.__version__)
print('A: ', tf.test.is_built_with_cuda)
print('B: ', tf.test.gpu_device_name())
local_device_protos = device_lib.list_local_devices()
print([x.name for x in local_device_protos if x.device_type == 'GPU'],
      [x.name for x in local_device_protos if x.device_type == 'CPU'])

# physical_devices = tf.config.list_physical_devices('GPU')
# for device in physical_devices:
#     tf.config.experimental.set_memory_growth(device, True)

memSize = 40000
batch_size = 10
learningRate = 0.001
gamma = 0.9
epsilon = 1.
epsilonDecayRate = 0.0002
min_epsilon = 0.05
filepathToSave = 'model.h5'
mapSize = 20
actions_count = len(Constants.actions)
input_state_layers = 11
nn_predict = NN((mapSize, mapSize, input_state_layers), learningRate)
model_predict = nn_predict.model

nn_target = NN((mapSize, mapSize, input_state_layers), learningRate)
model_target = nn_target.model


maxEpochs = 500000

maxSteps = 200
scores = []
total_reward = 0
total_steps = 0
max_reward = 0
bots_count = 2
copy_weights_init = 500
copy_weights = copy_weights_init
plot_x = []
plot_y = []

q = Q(memSize, gamma)
game = Game()
bots = []
for i in range(bots_count):
    bots.append(Bot('player_' + str(i)))
    game.signup_bot(bots[i])

game.set_owner(bots[0])
bot = bots[0]
opponent = TestBot(game, None, bots[1], True)
game.create()
for i in range(1, bots_count):
    game.connect_bot(bots[i])

epoch = 0
for epoch in range(maxEpochs):
    gameOver = False
    step = 0
    placed = 0
    game.start()
    while not gameOver:
        current_state = bot.set_state(game.get_objects(), mapSize)
        start_objects = bot.map_objects
        for map_object in start_objects:
            copy_weights -= 1
            step += 1
            total_steps += 1

            global_state, local_state = bot.get_padded_state(map_object)
            action = bot.predict_action(model_predict, map_object, epsilon, global_state, local_state)
            response = game.post_actions_and_take_turn(action, bot)
            current_state = bot.set_state(game.get_objects(), mapSize)
            global_state_next, local_state_next = bot.get_padded_state(map_object)
            invalidActions = response['invalidActions']



            # Calculate reward for placed settlement
            if action[2] == 8 and invalidActions == 0:
                reward = 0.1
                gameOver = True
                if current_state[19, 19, 0] > 3:
                    reward = 1
                    placed = step
            else:
                reward = -0.2
            if invalidActions == 1:
                reward = -1



            q.remember([[global_state, local_state], action, reward, [global_state_next, local_state_next]], gameOver)
            q.train_on_batch(model_predict, model_target, batch_size)

            print("epoch: ", epoch, "step: ", step, "epsilon: ", epsilon)
            print("action: ", action, reward)

            if epsilon > min_epsilon:
                epsilon -= epsilonDecayRate

            if step > maxSteps:
                gameOver = True

            if copy_weights == 0:
                copy_weights = copy_weights_init
                model_target.set_weights(model_predict.get_weights())

            total_reward += reward
            if reward > max_reward:
                max_reward = reward
                model_predict.save(filepathToSave)

            if total_steps % 50 == 0:
                scores.append(total_reward)
                total_reward = 0
                # plot_x.append(epoch)
                # plot_y.append(step)
                plt.plot(scores)
                plt.xlabel('Steps / 50')
                plt.ylabel('Reward ')
                plt.savefig('stats.png')
                plt.close()
                model_predict.save(filepathToSave)
        game.end_turn(bot)
        opponent.play()
