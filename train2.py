# Importowanie bibliotek
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

memSize = 6000
batchSize = 8
learningRate = 0.0001
gamma = 0.9
nLastStates = 4
epsilon = 1.
epsilonDecayRate = 0.002
minEpsilon = 0.05

filepathToSave = 'model2.h5'
mapSize = 20
actions_count = len(Constants.actions)
brain = NN((mapSize, mapSize, 11), learningRate)
model = brain.model
dqn = Dqn(memSize, gamma)

epoch = 0
maxEpochs = 500
step = 0
maxSteps = 100
scores = []
maxReward = 0
nCollected = 0.
total_reward = 0
bots_count = 1

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
    game.start()
    currentState = bot.set_state(game.get_objects(), mapSize)
    gameOver = False
    step = 0

    while not gameOver and step < maxSteps:
        step += 1

        if np.random.rand() < epsilon:
            qvalues = np.random.rand(mapSize, mapSize, actions_count)
        else:
            qvalues = model.predict(np.expand_dims(currentState, axis=0))[0]

        actions = bot.map_action(qvalues)

        response = game.post_actions_and_take_turn(actions, bot)
        invalidActions = response['invalidActions']
        nextState = bot.set_state(game.get_objects(), mapSize)
        # reward = bot.calculate_reward(invalidActions)
        if invalidActions == 1:
            reward = -1
        else:
            reward = 1
        print(reward, epsilon)
        gameOver = False  # todo

        dqn.remember([currentState, actions, reward, nextState], gameOver)
        inputs, targets = dqn.get_batch(model, batchSize)
        model.train_on_batch(inputs, targets)

        currentState = nextState

        total_reward += reward
        if reward > maxReward:
            maxReward = reward
            model.save(filepathToSave)

        if epsilon > minEpsilon:
            epsilon -= epsilonDecayRate

        if step % 100 == 0 and epoch != 0:
            scores.append(total_reward / 10)
            total_reward = 0
            plt.plot(scores)
            plt.xlabel('Epoch / 100')
            plt.ylabel('Average Score')
            plt.savefig('stats.png')
            plt.close()

    print('Epoch: ' + str(epoch) + ' Current Best: ' + str(maxReward) + ' Epsilon: {:.5f}'.format(epsilon))
