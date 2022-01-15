import gym
import gym.envs.toy_text.frozen_lake as fl
import numpy as np
import pdb

SEED = 12345

np.set_printoptions(precision=3)

# Create environment
env = fl.FrozenLakeEnv()  # gym.make('FrozenLake-v1')
env.seed(SEED)

# It should print a rendering of the FrozenLake environment
env.render()
pdb.set_trace(header="\n*** *** ***\nHello, you entered the interactive source code debugger.\nHere you can "
                     "experiment with elements "
                     "that have been defined. For instance you can call again env.render().\nTo continue, just type "
                     "continue.\nTo quit the program, just type quit.\n*** *** ***\n")

# Of course, you should RTFM (please do a quick peek into the doc)
help(fl)
# ... and again
help(gym.envs.toy_text.discrete)
pdb.set_trace(header="\nTake some time to read the doc.\n")

# Some printing to understand the 'env' class
print("Action Space - ", env.action_space, env.nA)
print("State Space - ", env.observation_space, env.nS)
# isd is the initial state distribution
print("\nInitial State Distribution  - ", env.isd, np.sum(env.isd) == 1)
print("\nTransitions from STATE 1 when going LEFT\n", env.P[1][0])
pdb.set_trace(header="\nTake some time to understand the few previous lines.\nDon't forget that you can play a little "
                     "with the variables (but be careful to not modify them).\n")


# If you understand those two functions
# you understand how states are linked to env.render()
def to_s(row, col):
    return row * env.ncol + col


def to_row_col(s):
    col = s % env.ncol
    row = s // env.ncol
    return row, col


# Documentation for interacting with a gym environment
help(gym.Env)
pdb.set_trace(header="\nRemember to read the doc if you don't understand/recall something.\n")
