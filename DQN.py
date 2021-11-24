import numpy as np


class Dqn(object):

    def __init__(self, max_memory=100, discount=0.9):
        self.memory = list()
        self.max_memory = max_memory
        self.discount = discount

    def remember(self, transition, game_over):
        self.memory.append([transition, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)

        inputs = []
        for i in range(min(len_memory, batch_size)):
            inputs.append(np.zeros_like(self.memory[0][0][0]))

        targets = np.zeros((min(len_memory, batch_size), model.output_shape[1], model.output_shape[2], model.output_shape[3]))

        for i, idx in enumerate(np.random.randint(0, len_memory, size=min(len_memory, batch_size))):
            current_state, actions, reward, next_state = self.memory[idx][0]
            game_over = self.memory[idx][1]
            inputs[i] = current_state
            targets[i] = model.predict(np.expand_dims(current_state, axis=0))[0]
            next_predicted = model.predict(np.expand_dims(next_state, axis=0))[0]
            for a in actions:
                q_next = np.max(next_predicted[a[0], a[1]])
                if game_over:
                    targets[i][a[0], a[1], [2]] = reward
                else:
                    targets[i][a[0], a[1], [2]] = reward + self.discount * q_next
        return np.array(inputs), targets
