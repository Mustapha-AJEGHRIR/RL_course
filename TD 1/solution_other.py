# %%
import gym.envs.toy_text.frozen_lake as fl
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
 
from numpy.core.numeric import identity
 
np.set_printoptions(precision=3)
NBR_EPISODES = 50000
HORIZON = 200
GAMMA = 0.9
SEED = int("whatAnAwesomePracticalSession", base=36) % 2 ** 31
 
# Create environment
env = fl.FrozenLakeEnv()  # gym.make('FrozenLake-v1')
env.seed(SEED)
 
 
def to_s(row, col):
    return row * env.ncol + col
 
 
def to_row_col(s):
    col = s % env.ncol
    row = s // env.ncol
    return row, col
 
 
def print_values(v):
    for row in range(env.nrow):
        for col in range(env.ncol):
            s = f"{v[to_s(row, col)]:.3}"
            print(s, end=" " * (8 - len(s)))
        print("")
 
 
def convert_time(t1, t2):
    return f"Running time: {t2 - t1:.4f} sec\n"
 
 
# %%
print("\n\n######################")
print("##### Question 4 #####")
print("######################\n")
print("EXPECTED VALUE METHOD\n")
 
 
def value_function_expected(pi):
    """
    Calculate value function of a policy pi using Monte Carlo method
    """
    V_pi = np.zeros(env.nS)
    for s_initial in range(env.nS):
        env.isd = np.zeros(env.nS)
        env.isd[s_initial] = 1
        for _ in range(NBR_EPISODES):
            env.reset()
            expected_reward = 0
            s = s_initial
            for t in range(HORIZON):
                s, r, done, _ = env.step(pi[s])
                expected_reward += r * (GAMMA ** t)
                if done:
                    break
            V_pi[s_initial] += expected_reward
    V_pi = V_pi / NBR_EPISODES
    return V_pi
 
 
simple_pi = fl.RIGHT * np.ones(env.nS)
starting_time = perf_counter()
V_simple_pi = value_function_expected(simple_pi)
print(convert_time(starting_time, perf_counter()))
print(f"Value estimate of the starting point: {V_simple_pi[0]:.3f}")
print(f"Value function of the always RIGHT policy:\n")
print_values(V_simple_pi)
 
# reset the original isd
env.isd = np.zeros(env.nS)
env.isd[0] = 1
 
# %%
# pdb.set_trace()
# Question 5
print("\n######################")
print("##### Question 5 #####")
print("######################\n")
print("LINEAR SYSTEM METHOD\n")
 
# Policy Iteration: the Policy Evaluation Step (slide 48)
# https://towardsdatascience.com/getting-started-with-markov-decision-processes-reinforcement-learning-ada7b4572ffb
def value_function(pi):
    """
    Calculate value function of a policy pi using Linear System method
 
    pi : int array
    For each index i, pi[i] is the action (int) chosen in state i
 
    return:
    ------
    V_pi : float array
    For each index i, V_pi[i] is the value (float) of the state i
    """
    R_pi = np.zeros(env.nS)
    P_pi = np.zeros((env.nS, env.nS))
    for s in range(env.nS):
        for p, s_, r, _ in env.P[s][pi[s]]:  # iterate over all possible transitions
            R_pi[s] += r * p  # R_pi[s] = sum(r * p)
            P_pi[s][s_] += p  # P_pi[s][s_] = sum(p)
    I = np.eye(env.nS)
    V_pi = np.linalg.solve(I - GAMMA * P_pi, R_pi)
    return V_pi
 
 
simple_pi = fl.RIGHT * np.ones(env.nS)
starting_time = perf_counter()
V_simple_pi = value_function(simple_pi)
print(convert_time(starting_time, perf_counter()))
print(f"Value estimate of the starting point: {V_simple_pi[0]:.3f}")
print(f"Value function of the always RIGHT policy:\n")
print_values(V_simple_pi)
 
# %%
# pdb.set_trace()
# Question 6
print("\n######################")
print("##### Question 6 #####")
print("######################\n")
print("BELLMAN OPERATOR METHOD\n")
 
# https://towardsdatascience.com/planning-by-dynamic-programming-reinforcement-learning-ed4924bbaa4c
# https://rll.berkeley.edu/deeprlcourse-fa15/docs/mdp-cheatsheet.pdf
def value_function_2(pi, epsilon, max_iter):
    """ 
    Calculate value function of a policy pi using Bellman Operator method
 
    pi : int array
    For each index i, pi[i] is the action (int) chosen in state i
 
    epsilon : float
    Used as a threshold for the stopping rule
 
    max_iter : int
    Hard threshold on the number of loops
 
    return:
    ------
    V_pi : float array
    For each index i, V_pi[i] is the value (float) of the state i
    """
    # Compute both the reward vector R_pi and
    # transition matrix P_pi associated to the policy on the given env
    R_pi = np.zeros(env.nS)
    P_pi = np.zeros((env.nS, env.nS))
    for s in range(env.nS):
        for p, s_, r, _ in env.P[s][pi[s]]:  # iterate over all possible transitions
            R_pi[s] += r * p  # R_pi[s] = sum(r * p)
            P_pi[s][s_] += p  # P_pi[s][s_] = sum(p)
 
    V_pi = np.zeros(env.nS)
    delta_inf = np.array([])
    for i in range(max_iter):
        V_pi_old = V_pi.copy()
        V_pi = R_pi + GAMMA * P_pi @ V_pi_old  # Bellman Operator
        delta_inf = np.append(delta_inf, np.max(np.abs(V_pi - V_pi_old)))
        if delta_inf[-1] < epsilon:
            break
 
    return V_pi, delta_inf
 
 
starting_time = perf_counter()
V_simple_pi, Delta_inf = value_function_2(simple_pi, 1e-4, 10000)
print(convert_time(starting_time, perf_counter()))
print(f"Value function of the always RIGHT policy:\n")
print_values(V_simple_pi)
 
plt.figure()
plt.title(
    "Semi-log graph of $n \mapsto || V_{n+1} - V_n ||_\infty $ \n\
The Linearity of this graph proves exponential convergence"
)
plt.semilogy(Delta_inf)
plt.xlabel("Iterate")
plt.ylabel(r"$|| V_{n+1} - V_n ||_\infty$")
plt.savefig("question6.png")
print(f"\nNumber of iterations: {Delta_inf.size}")
print(f"Last residual {Delta_inf[-1]:.6f}")
 
# %%
# pdb.set_trace()
# Question 7
print("\n######################")
print("##### Question 7 #####")
print("######################\n")
print("OPTIMAL BELLMAN OPERATOR\n")
 
 
def value_function_optimal(epsilon, max_iter):
    """ Calculate value function of a policy pi using Optimal Bellman Operator method
    epsilon : float
    Used as a threshold for the stopping rule
 
    max_iter : int
    Hard threshold on the number of loops
 
    returns:
    -------
    V_opt : float array, (env.nS,) size
    Optimal value function on the FrozenLake MDP given a discount GAMMA
    V_opt[state index] = Value of that state
    """
    R = np.zeros((env.nS, env.nA))
    P = np.zeros((env.nS, env.nA, env.nS))
    for s in range(env.nS):
        for a in range(env.nA):
            for p, s_, r, _ in env.P[s][a]:
                R[s, a] += r * p
                P[s, a, s_] += p
 
    V_opt = np.zeros(env.nS)
    delta_inf = np.array([])
    for i in range(max_iter):
        V_opt_old = V_opt.copy()
        V_opt = np.max(R + GAMMA * P @ V_opt_old, axis=1)
        delta_inf = np.append(delta_inf, np.max(np.abs(V_opt - V_opt_old)))
        if delta_inf[-1] < epsilon:
            break
    return V_opt, delta_inf
 
 
starting_time = perf_counter()
V_opt, Delta_inf = value_function_optimal(1e-4, 10000)
print(convert_time(starting_time, perf_counter()))
print(f"Optimal value function:\n")
print_values(V_opt)
 
plt.figure()
plt.title(
    "Semi-log graph of $n \mapsto || V_{n+1} - V_n ||_\infty $ \n\
The Linearity of this graph proves exponential convergence"
)
plt.semilogy(Delta_inf)
plt.xlabel("Iterate")
plt.ylabel(r"$|| V_{n+1} - V_n ||_\infty$")
plt.savefig("question7.png")
print(f"\nNumber of iterations: {Delta_inf.size}")
print(f"Last residual {Delta_inf[-1]:.6f}")
 
# %%
# pdb.set_trace()
# Question 8
print("\n######################")
print("##### Question 8 #####")
print("######################\n")
print("VALUE ITERATION\n")
 
 
def value_iteration(epsilon, max_iter):
    """
    epsilon : float
    Used as a threshold for the stopping rule
 
    max_iter : int
    Hard threshold on the number of loops
 
    returns:
    -------
    Pi_opt : int array, size (env.nS,)
    An optimal policy
    """
    R = np.zeros((env.nS, env.nA))
    P = np.zeros((env.nS, env.nA, env.nS))
    for s in range(env.nS):
        for a in range(env.nA):
            for p, s_, r, _ in env.P[s][a]:
                R[s, a] += r * p
                P[s, a, s_] += p
 
    V_opt = np.zeros(env.nS)
    Pi_opt = np.zeros(env.nS, dtype=int)
    delta_inf = np.array([])
    for i in range(max_iter):
        V_opt_old = V_opt.copy()
        # calculate R + GAMMA * P @ V_opt_old
        action_values = R + GAMMA * P @ V_opt_old
        V_opt = np.max(action_values, axis=1)
        Pi_opt = np.argmax(action_values, axis=1)
        delta_inf = np.append(delta_inf, np.max(np.abs(V_opt - V_opt_old)))
        if delta_inf[-1] < epsilon:
            break
    return Pi_opt
 
 
ARROWS = {fl.RIGHT: "→", fl.LEFT: "←", fl.UP: "↑", fl.DOWN: "↓"}
 
 
def print_policy(pi):
    for row in range(env.nrow):
        for col in range(env.ncol):
            print(ARROWS[pi[to_s(row, col)]], end="")
        print("")
 
 
starting_time = perf_counter()
Pi_opt = value_iteration(1e-4, 1000)
print(convert_time(starting_time, perf_counter()))
print("An optimal policy is:\n")
print_policy(Pi_opt)
 
# %%
# pdb.set_trace()
# Question 9
print("\n######################")
print("##### Question 9 #####")
print("######################\n")
print("POLICY ITERATION\n")
 
 
# The danger of Policy Iteration lies in the stopping criterion
# If not careful, one might end up with an algorithm that does not
# terminate and oscillates between optimal policies
# Even if it is computationally more expensive, we sometimes rather
# compare value functions of the policies than policies from one iterate
# to another.
 
# An easy improvement on the following code would be to use
# a warm start for policy evaluation steps (if iteration methods is used)
# That is to say, using the previously computed value function
# as the first step for the next policy evaluation
 
 
def policy_improvement(V, Pi):
    """
    V : float array, size (env.nS,)
    Value function of a policy
 
    returns:
    -------
    Pi : int array, size (env.nS,)
    A policy that is greedy with respect to V
 
    policy_stable : bool
    True if the policy is stable, i.e. the policy is always optimal
    """
    R = np.zeros((env.nS, env.nA))
    P = np.zeros((env.nS, env.nA, env.nS))
    for s in range(env.nS):
        for a in range(env.nA):
            for p, s_, r, _ in env.P[s][a]:
                R[s, a] += r * p
                P[s, a, s_] += p
 
    for s in range(env.nS):
        a = Pi[s]
        Pi[s] = np.argmax(R[s, :] + GAMMA * P[s, :, :] @ V)
        if a != Pi[s]:
            return Pi, False  # policy is not stable
 
    return Pi, True  # policy is stable
 
 
def policy_iteration(epsilon, max_iter):
    """
    epsilon : float
    Used as a threshold for the stopping rule
 
    max_iter : int
    Hard threshold on the number of loops
 
    returns:
    -------
    Pi : int array, size (env.nS,)
    An optimal policy
    """
    V_opt = np.zeros(env.nS)
    Pi_opt = np.zeros(env.nS, dtype=int)
 
    for i in range(max_iter):
        # policy evaluation
        V_opt, _ = value_function_2(Pi_opt, epsilon, max_iter)
 
        # policy improvement
        Pi_opt, policy_stable = policy_improvement(V_opt, Pi_opt)
 
        if policy_stable:
            return Pi_opt
 
    return Pi_opt
 
 
starting_time = perf_counter()
Pi_opt = policy_iteration(10e-4, 1000)
print(convert_time(starting_time, perf_counter()))
print("An optimal policy is:\n")
print_policy(Pi_opt)
 
# %%
# Question 10
 
print("\n#######################")
print("##### Question 10 #####")
print("#######################\n")
print("COMPARE THE TWO METHODS\n")
print(
    """
In value iteration, we start with a fixed value function, whereas in
policy iteration, we start with a fixed policy. In both cases, we
iterate until convergence.
 
The policy iteration method is more efficient than the value iteration
method because it does not require the computation of the value function
at each iteration.
"""
)
 
# %%
# pdb.set_trace()
# Question 11
print("\n#######################")
print("##### Question 11 #####")
print("#######################\n")
print("OPTIMAL Q-BELLMAN OPERATOR METHOD\n")
 
 
def state_value_function_optimal(epsilon, max_iter):
    """
    epsilon : float
    Used as a threshold for the stopping rule
 
    max_iter : int
    Hard threshold on the number of loops
 
    returns:
    -------
    q_opt : float array, (env.nS, env.nA) size
    Optimal state-action value function on the FrozenLake MDP
    given a discount GAMMA
    q_opt[state index][action index] = state-action value of that state
    """
 
    R = np.zeros((env.nS, env.nA))
    P = np.zeros((env.nS, env.nA, env.nS))
    for s in range(env.nS):
        for a in range(env.nA):
            for p, s_, r, _ in env.P[s][a]:
                R[s, a] += r * p
                P[s, a, s_] += p
 
    q_opt = np.zeros((env.nS, env.nA))
    delta_inf = np.array([])
    for i in range(max_iter):
        # q_opt[s, a] = np.sum([p * (r + GAMMA * q_opt_old[s_, a_]) for p, s_, r, _ in env.P[s][a]])
        q_opt_old = q_opt.copy()
        q_opt = R + GAMMA * P @ np.max(q_opt_old, axis=1)
 
        delta_inf = np.append(delta_inf, np.max(np.abs(q_opt - q_opt_old)))
        if delta_inf[-1] < epsilon:
            break
 
    return q_opt, delta_inf
 
 
starting_time = perf_counter()
Q_opt, Delta_inf = state_value_function_optimal(1e-4, 100)
print(convert_time(starting_time, perf_counter()))
# print(Q_opt)
V_opt = np.max(Q_opt, axis=1)
print(f"Optimal value function:\n")
print_values(V_opt)
 
plt.figure()
plt.title(
    "Semi-log graph of $n \mapsto || Q_{n+1} - Q_n ||_\infty $ \n\
The Linearity of this graph proves exponential convergence"
)
plt.semilogy(Delta_inf)
plt.xlabel("Iterate")
plt.ylabel(r"$|| Q_{n+1} - Q_n ||_\infty$")
plt.savefig("question11.png")
print(f"\nNumber of iterations: {Delta_inf.size}")
print(f"Last residual {Delta_inf[-1]:.6f}")
 
# %%
# Question 12
print("\n#######################")
print("##### Question 12 #####")
print("#######################\n")
 
Pi_opt = np.argmax(Q_opt, axis=1)
print("\nAn optimal policy is:\n")
print_policy(Pi_opt)
 
# %%
 
# Question 13
print("\n#######################")
print("##### Question 13 #####")
print("#######################\n")
print("RENDER A TRAJECTORY\n")
 
 
# render policy
def trajectory(pi, max_moves=20):
    done = False
    i = 0
    env.reset()
    env.reset()
    cumulative_reward = 0
    discount = 1
    while not done and i < max_moves:
        i += 1
        _, r, done, _ = env.step(pi[env.s])
        cumulative_reward += discount * r
        discount *= GAMMA
        env.render()
        print("")
    return cumulative_reward
 
 
cr = trajectory(Pi_opt)
print("\nThe GOAL has been reached! Congrats! :-)")
print(f"The cumulative discounted reward along the above trajectory is: {cr:.3f}\n")