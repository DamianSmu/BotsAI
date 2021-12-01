import os
import Constants
from Bot import Bot
from Game import Game
from NN import NN
from DQN import Dqn
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.client import device_lib

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
batchSize = 10
learningRate = 0.001
gamma = 0.9
epsilon = 1.
epsilonDecayRate = 0.0002
minEpsilon = 0.05
filepathToSave = 'model.h5'
mapSize = 20
actions_count = len(Constants.actions)

nn_predict = NN((mapSize, mapSize, 11), learningRate)
model_predict = nn_predict.model

nn_target = NN((mapSize, mapSize, 11), learningRate)
model_target = nn_target.model

dqn = Dqn(memSize, gamma)
epoch = 0
maxEpochs = 500000
step = 0
maxSteps = 20
scores = []
bots_count = 1
copy_weights_init = 500
copy_weights = copy_weights_init
plot_x = []
plot_y = []

game = Game()
bots = []
for i in range(bots_count):
    bots.append(Bot('player_' + str(i)))
    game.signup_bot(bots[i])

game.set_owner(bots[0])
bot = bots[0]
game.create()
for i in range(1, bots_count):
    game.connect_bot(bots[i])

for epoch in range(maxEpochs):
    gameOver = False
    step = 0
    placed = 0
    game.start()
    currentState = bot.set_state(game.get_objects(), mapSize)
    while not gameOver:
        copy_weights -= 1
        step += 1
        if np.random.rand() < epsilon:
            qvalues = np.random.rand(mapSize, mapSize, actions_count)
        else:
            qvalues = model_predict.predict(np.expand_dims(currentState, axis=0))[0]

        actions = bot.map_action(qvalues)

        response = game.post_actions_and_take_turn(actions, bot)
        invalidActions = response['invalidActions']
        nextState = bot.set_state(game.get_objects(), mapSize)
        # reward = bot.calculate_reward(invalidActions)

        # Calculate reward for placed settlement
        if actions[0][2] == 8 and invalidActions == 0:
            reward = 1
            gameOver = True
        elif invalidActions == 0 and actions[0][2] == 8 and currentState[actions[0][0], actions[0][1], 0] in [2, 3]:
            reward = 5
            gameOver = True
            placed = step
        else:
            reward = -0.2

        if invalidActions == 1:
            reward = -1

        dqn.remember([currentState, actions, reward, nextState], gameOver)
        inputs, targets = dqn.get_batch(model_predict, model_target, batchSize)
        print("epoch: ", epoch, "step: ", step, "epsilon: ", epsilon)
        print("action: ", actions[0], reward)
        model_predict.train_on_batch(inputs, targets)
        currentState = nextState

        # total_reward += reward
        # if reward > maxReward:
        #     maxReward = reward
        #     model.save(filepathToSave)

        if epsilon > minEpsilon:
            epsilon -= epsilonDecayRate

        if step > maxSteps:
            gameOver = True

        if copy_weights == 0:
            copy_weights = copy_weights_init
            model_target.set_weights(model_predict.get_weights())


        if placed != 0:
            #scores.append(placed)
            #total_reward = 0
            plot_x.append(epoch)
            plot_y.append(step)
            plt.scatter(plot_x, plot_y)
            plt.xlabel('Epoch')
            plt.ylabel('Step number')
            plt.savefig('stats.png')
            plt.close()
            model_predict.save(filepathToSave)
