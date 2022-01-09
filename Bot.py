import numpy as np
import Constants


class Bot:
    def __init__(self, name):
        self.name = name
        self.id = None
        self.token = None
        self.state = []
        self.terrain = None
        self.units = []
        self.settlements = []

    def set_state(self, objects, map_size=20):
        units = [x for x in objects if x['unit'] and x['unit']['type'] in ['SETTLERS', 'WARRIORS']]
        settlements = [x for x in objects if x['settlement'] and x['settlement']['type'] in ['SETTLEMENT']]

        my_units = [x for x in units if x and x['unit']['user']['id'] == self.id]
        my_settlements = [x for x in settlements if x and x['settlement']['user']['id'] == self.id]
        other_units = [x for x in units if x and x['unit']['user']['id'] != self.id]
        other_settlements = [x for x in settlements if x and x['settlement']['user']['id'] != self.id]

        self.units = []
        for u in my_units:
            self.units.append({'x': u['position']['x'], 'y': u['position']['y'], 'type': u['unit']['type']})

        self.settlements = []
        for u in my_settlements:
            self.settlements.append({'x': u['position']['x'], 'y': u['position']['y'], 'type': u['settlement']['type']})

        m = np.dstack([self.terrain, np.zeros((map_size, map_size, 6))])

        for o in my_units:
            m[o['position']['x'], o['position']['y'], 1] = o['unit']['defence']
            m[o['position']['x'], o['position']['y'], 2] = o['unit']['offence']

        for o in other_units:
            m[o['position']['x'], o['position']['y'], 3] = o['unit']['defence']
            m[o['position']['x'], o['position']['y'], 4] = o['unit']['offence']

        for o in my_settlements:
            m[o['position']['x'], o['position']['y'], 5] = o['settlement']['defence']

        for o in other_settlements:
            m[o['position']['x'], o['position']['y'], 6] = o['settlement']['defence']

        self.state = []
        for x in list(np.moveaxis(m, 2, 0)):
            self.state.append(np.pad(x, (map_size - 1, map_size - 1), "constant", constant_values=0))
        self.state = np.dstack(self.state)

        return self.state

    def get_padded_state(self, object, local_state_size=3, map_size=20):
        posx = object['x']
        posy = object['y']
        global_state = self.state[posx:posx + (2 * map_size) - 1, posy:posy + (2 * map_size) - 1]
        off = (global_state.shape[0] - local_state_size) // 2
        local_state = global_state[off:-off, off:-off]
        return global_state, local_state

    def predict_action(self, model, map_object, epsilon, global_state, local_state):
        if np.random.rand() < epsilon:
            qvalues = np.random.rand(len(Constants.actions))
        else:
            qvalues = model.predict([np.expand_dims(global_state, axis=0), np.expand_dims(local_state, axis=0)])[0]
        option = np.argmax(qvalues)

        return [map_object['x'],
                map_object['y'],
                option]

    # def calculate_reward(self, invalid_actions):
    #     partial_sum = np.sum(np.sum(self.state, axis=0), axis=0)
    #     partial_sum[0] *= 0
    #     partial_sum[1] *= 1
    #     partial_sum[2] *= 1
    #     partial_sum[3] *= 0
    #     partial_sum[4] *= 0
    #     partial_sum[5] *= 1
    #     partial_sum[6] *= 1
    #     partial_sum[7] *= 1
    #     partial_sum[8] *= 0
    #     partial_sum[9] *= 0
    #     partial_sum[10] *= 0
    #     return np.sum(partial_sum) - invalid_actions * 100
