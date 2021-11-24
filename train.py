import json
import os
import random

import numpy as np
from IPython.core.display import clear_output
import matplotlib.pyplot as plt

import NN
import tensorflow as tf
from Bot import Bot
from Game import Game

learning_rate = 0.01
gamma = 0.95
epsilon = 1.0
epsilon_final = 0.01
epsilon_decay = 0.998
last_reward = 0
W = 0
size = 10


def run(bot, reward, last_state, current_state):
    global epsilon, model, last_reward
    x = bot.set_state(current_state, size)

    if random.random() < epsilon:
        y = np.random.rand(size, size, 9)
    else:
        y = model.predict(np.asarray([x]))[0]

    actions, option = bot.map_action(y)
    if len(actions) == 0:
        print("")

    if bot.id in last_state:
        _x, _y, _option, _reward = last_state[bot.id]

        if reward[bot.id] > last_reward:
            r = 1
        elif reward[bot.id] < last_reward:
            r = -1
        else:
            r = 0

        for o in bot.objects:
            Q1 = _y[o['x']][o['y']][_option[o['x']][o['y']]]
            Q2 = y[o['x']][o['y']][_option[o['x']][o['y']]]
            v = r + gamma * (Q2 - Q1)
            _y[o['x']][o['y']][_option[o['x']][o['y']]] += learning_rate * v

        _y = y + learning_rate * _y
        _y_ = [_y]

        model.fit(np.asarray([_x]), np.asarray(_y_), epochs=1, verbose=1)
        if epsilon > epsilon_final:
            epsilon *= epsilon_decay

    last_state = [x, y, option, reward]
    return last_state, actions


episodes = 2
steps = 300

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

model = NN.get_model(10)
game = Game()

bots = []
rewards = []
bots_count = 2
for i in range(bots_count):
    bots.append(Bot('player_' + str(i)))
    game.signup_bot(bots[i])

game.set_owner(bots[0])

game.create()
for i in range(1, bots_count):
    game.connect_bot(bots[i])

log = []
for eps in range(episodes):
    print("######################### Episode {}".format(eps))
    game.start()
    last_state = {}
    reward = {}
    for step in range(100):
        current_state = game.get_objects()
        for bot in bots:
            print(chr(27) + "[2J")
            last_state[bot.id], actions = run(bot, reward, last_state, current_state)
            r = game.post_actions_and_take_turn(actions, bot)
            reward[bot.id] = bot.calculate_reward(r['invalidActions'])
            print(reward[bot.id])
            log.append([last_state[bot.id], reward[bot.id]])

log


model.save_weights("model_%d.h5")
