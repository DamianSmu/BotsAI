import numpy as np


class Q:
    def __init__(self, max_memory=5000, discount=0.9):
        self.memory = list()
        self.max_memory = max_memory
        self.discount = discount

    def save(self, transition, game_over):
        self.memory.append([transition, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, predict_net, target_net, batch_size=10):
        len_memory = len(self.memory)
        inputs_local = []
        inputs_global = []
        targets = []
        for i, idx in enumerate(np.random.choice(range(len_memory), min(batch_size, len_memory), replace=False)):
            state, action, reward, n_state = self.memory[idx][0]
            game_over = self.memory[idx][1]
            inputs_global.append(state[0])
            inputs_local.append(state[1])
            targets.append(predict_net.predict([np.expand_dims(state[0], axis=0), np.expand_dims(state[1], axis=0)])[0])
            next_predicted = target_net.predict([np.expand_dims(n_state[0], axis=0), np.expand_dims(n_state[1], axis=0)])[0]
            if game_over:
                targets[i][action[2]] = reward
            else:
                targets[i][action[2]] = reward + self.discount * np.max(next_predicted)
        return [np.asarray(inputs_global), np.asarray(inputs_local)], np.asarray(targets)

    def train_on_batch(self, predict_net, target_net, batch_size=10):
        inputs, targets = self.get_batch(predict_net, target_net, batch_size)
        return predict_net.train_on_batch(inputs, targets)
