import numpy as np


class Q:
    def __init__(self, max_memory=50000, discount=0.9):
        self.memory = list()
        self.max_memory = max_memory
        self.discount = discount

    def remember(self, transition, game_over):
        self.memory.append([transition, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, predict_network, target_network, batch_size=10):
        len_memory = len(self.memory)
        inputs_local = []
        inputs_global = []
        targets = []
        for i, idx in enumerate(np.random.randint(0, len_memory, size=min(len_memory, batch_size))):
            current_state, action, reward, next_state = self.memory[idx][0]
            game_over = self.memory[idx][1]
            inputs_global.append(current_state[0])
            inputs_local.append(current_state[1])
            targets.append(predict_network.predict([np.expand_dims(current_state[0], axis=0), np.expand_dims(current_state[1], axis=0)])[0])
            next_predicted = target_network.predict([np.expand_dims(next_state[0], axis=0), np.expand_dims(next_state[1], axis=0)])[0]
            q_next = np.max(next_predicted)
            if game_over:
                targets[i][action[2]] = reward
            else:
                targets[i][action[2]] = reward + self.discount * q_next
        return [np.asarray(inputs_global), np.asarray(inputs_local)], np.asarray(targets)

    def train_on_batch(self, predict_network, target_network, batch_size=10):
        inputs, targets = self.get_batch(predict_network, target_network, batch_size)
        predict_network.train_on_batch(inputs, targets)

