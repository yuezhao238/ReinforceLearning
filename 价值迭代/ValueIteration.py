import numpy as np


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
                delta = max(delta, abs(new_value - value[i, j]))
                value[i, j] = new_value
        if delta / (4 * 12) < 1:
            break


def policy_improvement(value, policy):
    policy_stable = True
    for i in range(4):
        for j in range(12):
            if get_state(i, j) == 'trap' or get_state(i, j) == 'terminal':
                continue
            old_action = policy[(i, j)]
            action_list = ['up', 'down', 'left', 'right']
            action_value = []
            for action in action_list:
                row, col, reward = get_next_state(i, j, action)
                action_value.append(reward + value[row, col])
            policy[(i, j)] = action_list[np.argmax(action_value)]
            if old_action != policy[(i, j)]:
                policy_stable = False
    return policy_stable


def value_iteration(value, policy):
    while True:
        policy_evaluation(value, policy)
        if policy_improvement(value, policy):
            break


def print_policy(policy):
    for i in range(4):
        for j in range(12):
            if get_state(i, j) == 'trap' or get_state(i, j) == 'terminal':
                print(get_state(i, j), end='\t')
            else:
                print(policy[(i, j)], end='\t')
        print()


def print_value(value):
    for i in range(4):
        for j in range(12):
            if get_state(i, j) == 'trap' or get_state(i, j) == 'terminal':
                print(get_state(i, j), end='\t')
            else:
                print('%.2f' % value[i, j], end='\t')
        print()


def main():
    value = init_value()
    policy = init_policy()
    value_iteration(value, policy)
    print_policy(policy)
    print_value(value)


if __name__ == '__main__':
    main()
