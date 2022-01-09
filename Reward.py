import numpy as np


def find_closest_resource(state, state_next):
    res_list = np.argwhere((state[:, :, 0] == 10) & (state[:, :, 6] == 0))
    res_list_next = np.argwhere((state_next[:, :, 0] == 10) & (state_next[:, :, 6] == 0))
    x, y = tuple(np.argwhere(state[:, :, 1] > 0)[0])
    xn, yn = tuple(np.argwhere(state_next[:, :, 1] > 0)[0])
    closest = 40
    for r in res_list:
        xr, yr = r
        if abs(x - xr) + abs(y - yr) < closest:
            closest = abs(x - xr) + abs(y - yr)
    closest_next = 40
    for r in res_list_next:
        xr, yr = r
        if abs(xn - xr) + abs(yn - yr) < closest_next:
            closest_next = abs(xn - xr) + abs(yn - yr)
    return (closest - closest_next) / 4


def find_closest_opponent(state, state_next):
    op_list = np.argwhere((state[:, :, 4]) == 1)
    op_list_next = np.argwhere((state_next[:, :, 4]) == 1)
    pos = np.argwhere(state[:, :, 1] > 0)
    if len(pos) == 0:
        return 0
    x, y = tuple(pos[0])
    pos_next = np.argwhere(state_next[:, :, 1] > 0)
    if len(pos_next) == 0:
        return 0
    xn, yn = tuple(pos_next[0])
    closest = 40
    for r in op_list:
        xr, yr = r
        if abs(x - xr) + abs(y - yr) < closest:
            closest = abs(x - xr) + abs(y - yr)
    closest_next = 40
    for r in op_list_next:
        xr, yr = r
        if abs(xn - xr) + abs(yn - yr) < closest_next:
            closest_next = abs(xn - xr) + abs(yn - yr)
    return (closest - closest_next) / 4


def settlements_strategy_reward(current_state, current_state_next, local_state, action, invalid_actions):
    if invalid_actions == 0:
        reward_for_distance = find_closest_resource(state=current_state, state_next=current_state_next)
        if action[2] == 8:
            reward = 0.5
            if local_state[1, 1, 0] > 3:
                reward = 5
        else:
            reward = reward_for_distance
    else:
        reward = -1
    return reward


def attack_strategy_reward(current_state, current_state_next, local_state, action, invalid_actions):
    if invalid_actions == 0:
        reward_for_distance = find_closest_opponent(state=current_state, state_next=current_state_next)
        if action[2] in [4, 5, 6, 7]:
            reward = 5
        else:
            reward = reward_for_distance
    else:
        if action[2] in [4, 5, 6, 7]:
            reward = -0.1
        else:
            reward = -1
    return reward
