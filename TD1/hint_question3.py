import gym.envs.toy_text.frozen_lake as fl
import numpy as np
import matplotlib.pyplot as plt

SEED = int("whatAnAwesomePracticalSession", base=36) % 2**31
NBR_EPISODES = 100000
HORIZON = 200
GAMMA = 0.9

# Create environment
env = fl.FrozenLakeEnv()  # gym.make('FrozenLake-v1')
env.seed(SEED)

# First interaction with the environment
VALUE_START = np.zeros(NBR_EPISODES)
for i in range(NBR_EPISODES):
    env.reset()
    done = False
    t = 0
    discount = 1
    while (not done) and (t < HORIZON):
        next_state, r, done, _ = env.step(fl.RIGHT)
        VALUE_START[i] += discount * r
        discount *= GAMMA
        t += 1

print(f"Value estimate of the starting point: {np.mean(VALUE_START):.4f}")

offset = 10
plt.figure()
plt.title("Convergence of the Monte Carlo estimation\nof the value of the \
starting point")
plt.plot((np.cumsum(VALUE_START) / (np.arange(NBR_EPISODES) + 1))[offset:])
plt.show()
