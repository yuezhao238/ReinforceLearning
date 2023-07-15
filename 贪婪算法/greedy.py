import numpy as np

# get 10 probability machine
machine = np.random.rand(10)

# init machine choosed times
machine_choosed_times = np.zeros(10)

observation = np.zeros(10)


def action(states, round, epsilon=0.1):
    # explore
    if np.random.rand() < epsilon / np.exp(round / 1000):
        return np.random.randint(10)
    # exploit
    choose = np.argmax(states)
    return choose


def reward(action):
    # get reward
    return np.random.binomial(1, machine[action])


def update(action, reward):
    # update machine choosed times
    machine_choosed_times[action] += 1
    # update observation
    observation[action] = (observation[action] * (machine_choosed_times[action] - 1) + reward) / machine_choosed_times[action]


def get_states():
    # get states
    # return machine_choosed_times
    return observation


def circle():
    # expected reward of 1000 times
    print(10000 * np.max(machine))
    # circle
    total_reward = 0
    for i in range(10000):
        states = get_states()
        action_ = action(states, i)
        reward_ = reward(action_)
        update(action_, reward_)
        total_reward += reward_
    return total_reward


if __name__ == '__main__':
    print(circle())
