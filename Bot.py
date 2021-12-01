import numpy as np


class Bot:
    def __init__(self, name):
        self.name = name
        self.bots = {}
        self.id = None
        self.token = None
        self.state = None
        self.terrain = None
        self.objects = []

    def set_state(self, objects, map_size):
        units = [x for x in objects if x['mapObject']['type'] in ['SETTLERS', 'WARRIORS']]
        settlements = [x for x in objects if x['mapObject']['type'] in ['SETTLEMENT']]

        my_units = [x for x in units if x['mapObject']['user']['id'] == self.id]
        my_settlements = [x for x in settlements if x['mapObject']['user']['id'] == self.id]
        other_units = [x for x in units if x['mapObject']['user']['id'] != self.id]
        other_settlements = [x for x in settlements if x['mapObject']['user']['id'] != self.id]

        self.objects = []
        for u in my_units + my_settlements:
            self.objects.append({'x': u['position']['x'], 'y': u['position']['y'], 'type': u['mapObject']['type']})

        m_1 = np.zeros((map_size, map_size, 2))
        for o in my_units:
            m_1[o['position']['x'], o['position']['y'], 0] = o['mapObject']['defence']
            m_1[o['position']['x'], o['position']['y'], 1] = o['mapObject']['offence']

        m_2 = np.zeros((map_size, map_size, 2))
        for o in other_units:
            m_2[o['position']['x'], o['position']['y'], 0] = o['mapObject']['defence']
            m_2[o['position']['x'], o['position']['y'], 1] = o['mapObject']['offence']

        m_3 = np.zeros((map_size, map_size, 3))
        for o in my_settlements:
            m_3[o['position']['x'], o['position']['y'], 0] = o['mapObject']['defence']
            m_3[o['position']['x'], o['position']['y'], 1] = o['mapObject']['goldAmount']
            m_3[o['position']['x'], o['position']['y'], 2] = o['mapObject']['ironAmount']

        m_4 = np.zeros((map_size, map_size, 3))
        for o in other_settlements:
            m_4[o['position']['x'], o['position']['y'], 0] = o['mapObject']['defence']
            m_4[o['position']['x'], o['position']['y'], 1] = o['mapObject']['goldAmount']
            m_4[o['position']['x'], o['position']['y'], 2] = o['mapObject']['ironAmount']
        # 0: terr, a
        # 1: my_units_defence
        # 2: my_units_offence
        # 3: other_units_defence
        # 4: other_units_offence
        # 5: my_settlements_defence
        # 6: my_settlements_gold
        # 7: my_settlements_iron
        # 8: other_settlements_defence
        # 9: other_settlements_gold
        # 10: other_settlements_iron
        self.state = np.dstack([self.terrain, m_1, m_2, m_3, m_4])
        return self.state

    def calculate_reward(self, invalid_actions):
        partial_sum = np.sum(np.sum(self.state, axis=0), axis=0)
        partial_sum[0] *= 0
        partial_sum[1] *= 1
        partial_sum[2] *= 1
        partial_sum[3] *= 0
        partial_sum[4] *= 0
        partial_sum[5] *= 1
        partial_sum[6] *= 1
        partial_sum[7] *= 1
        partial_sum[8] *= 0
        partial_sum[9] *= 0
        partial_sum[10] *= 0
        return np.sum(partial_sum) - invalid_actions * 100

    def map_action(self, q_values):
        actions = []
        option = np.argmax(q_values, axis=2)

        for o in self.objects:
            d = [o['x'],
                 o['y'],
                 option[o['x']][o['y']]]
            print("q", q_values[o['x']][o['y']][option[o['x']][o['y']]])
            actions.append(d)
        return actions
