import numpy as np


# set up the environment
def get_state(row, col):
    if row != 3:
        return 'ground'
    if row == 3 and col == 0:
        return 'ground'
    if row == 3 and col == 11:
        return 'terminal'
    return 'trap'


def get_next_state(row, col, action):
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


def init_value():
    value = np.zeros((4, 12))
    for i in range(4):
        for j in range(12):
            if get_state(i, j) == 'trap':
                value[i, j] = -100
    return value


def init_policy():
    policy = {}
    for i in range(4):
        for j in range(12):
            if get_state(i, j) == 'trap':
                policy[(i, j)] = 'trap'
            elif get_state(i, j) == 'terminal':
                policy[(i, j)] = 'terminal'
            else:
                policy[(i, j)] = np.random.choice(['up', 'down', 'left', 'right'])
    return policy


def policy_evaluation(value, policy):
    while True:
        delta = 0
        for i in range(4):
            for j in range(12):
                if get_state(i, j) == 'trap' or get_state(i, j) == 'terminal':
                    continue
                row, col, reward = get_next_state(i, j, policy[(i, j)])
                new_value = reward + value[row, col]
                delta += abs(new_value - value[i, j])
                value[i, j] = new_value
        if delta / (4 * 12) < 1:
            break
    return value


def policy_improvement(value, policy):
    policy_stable = True
    for i in range(4):
        for j in range(12):
            if get_state(i, j) == 'trap' or get_state(i, j) == 'terminal':
                continue
            old_action = policy[(i, j)]
            row, col, reward = get_next_state(i, j, old_action)
            max_value = reward + value[row, col]
            best_action = old_action
            for action in ['up', 'down', 'left', 'right']:
                row, col, reward = get_next_state(i, j, action)
                new_value = reward + value[row, col]
                if new_value > max_value:
                    max_value = new_value
                    best_action = action
            policy[(i, j)] = best_action
            if old_action != best_action:
                policy_stable = False
    return policy, policy_stable


def policy_iteration():
    value = init_value()
    policy = init_policy()
    while True:
        value = policy_evaluation(value, policy)
        policy, policy_stable = policy_improvement(value, policy)
        if policy_stable:
            break
    return value, policy


def main():
    value, policy = policy_iteration()
    print('value:')
    print(value)
    print('policy:')
    print(policy)


if __name__ == '__main__':
    main()
