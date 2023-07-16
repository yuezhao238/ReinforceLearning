import numpy as np
import random


def get_state(row, col):
    if row != 3:
        return 'ground'
    if row == 3 and col == 0:
        return 'ground'
    if row == 3 and col == 11:
        return 'terminal'
    return 'trap'


def get_next_state(row, col, action):
    if get_state(row, col) in ['trap', 'terminal']:
        return row, col, 0
    if action == 'up':
        row -= 1
    elif action == 'down':
        row += 1
    elif action == 'left':
        col -= 1
    elif action == 'right':
        col += 1
    row = max(0, row)
    row = min(3, row)
    col = max(0, col)
    col = min(11, col)

    reward = -1
    if get_state(row, col) == 'trap':
        reward = -100

    return row, col, reward


def init_Q():
    return np.zeros((4, 12, 4))


Q = init_Q()


def get_action(row, col, epoch):
    if random.random() < 0.1 * np.exp(-epoch / 10):
        return random.choice(['up', 'down', 'left', 'right'])

    idx = np.argmax(Q[row, col])
    return ['up', 'down', 'left', 'right'][idx]


def TemporalDifference(row, col, action, reward, next_row, next_col, next_action):
    target = 0.9 * Q[next_row, next_col, ['up', 'down', 'left', 'right'].index(next_action)]
    target += reward

    value = Q[row, col, ['up', 'down', 'left', 'right'].index(action)]

    update = target - value
    update *= 0.1
    return update


def train():
    for epoch in range(1000):
        row = random.choice([0, 1, 2, 3])
        col = 0
        action = get_action(row, col, epoch)
        reward_sum = 0
        while get_state(row, col) not in ['terminal', 'trap']:
            next_row, next_col, reward = get_next_state(row, col, action)
            reward_sum += reward

            next_action = get_action(next_row, next_col, epoch)
            update = TemporalDifference(row, col, action, reward, next_row, next_col, next_action)
            Q[row, col, ['up', 'down', 'left', 'right'].index(action)] += update

            row = next_row
            col = next_col
            action = next_action

        if (epoch + 1) % 100 == 0:
            print('epoch: {}, reward_sum: {}'.format(epoch + 1, reward_sum))


train()
