import numpy as np


class Dqn(object):

    def __init__(self, max_memory=5000, discount=0.9):
        self.memory = list()
        self.max_memory = max_memory
        self.discount = discount

    def remember(self, transition, game_over):

        self.memory.append([transition, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, predict_network, target_network, batch_size=10):
        len_memory = len(self.memory)

        inputs = []
        for i in range(min(len_memory, batch_size)):
            inputs.append(np.zeros_like(self.memory[0][0][0]))

        targets = np.zeros(
            (min(len_memory, batch_size), predict_network.output_shape[1], predict_network.output_shape[2], predict_network.output_shape[3]))

        for i, idx in enumerate(np.random.randint(0, len_memory, size=min(len_memory, batch_size))):
            current_state, actions, reward, next_state = self.memory[idx][0
            ]
            game_over = self.memory[idx][1]
            inputs[i] = current_state
            targets[i] = predict_network.predict(np.expand_dims(current_state, axis=0))[0]
            next_predicted = target_network.predict(np.expand_dims(next_state, axis=0))[0]
            for a in actions:
                # q_next = np.max(next_predicted[a[0], a[1]])
                q_next = np.max(next_predicted[get_next_pos(a, next_state)])
                if game_over:
                    targets[i][a[0]][a[1]][a[2]] = reward
                else:
                    targets[i][a[0]][a[1]][a[2]] = reward + self.discount * q_next
        return np.array(inputs), targets


def get_next_pos(action, next_state):
    if action[2] == 0 and next_state[action[0], action[1], 1] == 0 and next_state[action[0], action[1] - 1, 1] != 0:
        return [action[0], action[1] - 1]
    if action[2] == 1 and next_state[action[0], action[1], 1] == 0 and next_state[action[0], action[1] + 1, 1] != 0:
        return [action[0], action[1] + 1]
    if action[2] == 2 and next_state[action[0], action[1], 1] == 0 and next_state[action[0] - 1, action[1], 1] != 0:
        return [action[0] - 1, action[1]]
    if action[2] == 3 and next_state[action[0], action[1], 1] == 0 and next_state[action[0] + 1, action[1], 1] != 0:
        return [action[0] + 1, action[1]]
    return [action[0], action[1]]
