# %%
import gym.envs.toy_text.frozen_lake as fl
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

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
            print(s, end=' ' * (8 - len(s)))
        print("")


def convert_time(t1, t2):
    return f"Running time: {t2 - t1:.4f} sec\n"

# %%
# Question 4
print("\n\n######################")
print("##### Question 4 #####")
print("######################\n")
print("EXPECTED VALUE METHOD\n")


def value_function_expected(pi):
    env = fl.FrozenLakeEnv()  # gym.make('FrozenLake-v1')
    env.seed(SEED)
    V = np.zeros(env.nS)
    for s in range(env.nS):
        isd = np.zeros(env.nS)
        isd[s] = 1
        env.isd = isd
        
        VALUE_START = np.zeros(NBR_EPISODES)
        for i in range(NBR_EPISODES):
            env.reset()
            done = False
            t = 0
            discount = 1
            while (not done) and (t < HORIZON):
                step = pi[env.s]
                next_state, r, done, _ = env.step(step)
                VALUE_START[i] += discount * r
                discount *= GAMMA
                t += 1
        V[s] = np.mean(VALUE_START)
    return V


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


def value_function(pi):
    """
    pi : int array
    For each index i, pi[i] is the action (int) chosen in state i

    return:
    ------
    V_pi : float array
    For each index i, V_pi[i] is the value (float) of the state i
    """
    P = np.zeros((env.nS, env.nS)) # Transition matrix, see slide 37 of the lecture "ShortLecture_MDPControl"
    m = np.zeros(env.nS)
    for state, state_possiblities in env.P.items():
        action = pi[state]
        for possibility in state_possiblities[action]: 
            proba, next_state, reward, done = possibility
            m[state] += reward * proba
            P[state, next_state] += proba

    return np.linalg.solve(np.eye(env.nS)- GAMMA * P, m)


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


def value_function_2(pi, epsilon, max_iter):
    """
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
    # Compute both the reward vector r_pi and
    # transition matrix P_pi associated to the policy on the given env
    V_pi = np.zeros(env.nS)
    
    P = np.zeros((env.nS, env.nS)) # Transition matrix, see slide 37 of the lecture "ShortLecture_MDPControl"
    m = np.zeros(env.nS)
    for state, state_possiblities in env.P.items():
        action = pi[state]
        for possibility in state_possiblities[action]: 
            proba, next_state, reward, done = possibility
            m[state] += reward * proba
            P[state, next_state] += proba

    delta_inf = []
    for _ in range(max_iter):
        V_old = V_pi
        V_pi = m + GAMMA * P @ V_old
        diff = np.max(np.abs(V_old - V_pi))
        delta_inf.append(diff)
        if diff < epsilon:
            break
        
    return V_pi, np.array(delta_inf)


starting_time = perf_counter()
V_simple_pi, Delta_inf = value_function_2(simple_pi, 1e-4, 10000)
print(convert_time(starting_time, perf_counter()))
print(f"Value function of the always RIGHT policy:\n")
print_values(V_simple_pi)

plt.figure()
plt.title("Semi-log graph of $n \mapsto || V_{n+1} - V_n ||_\infty $ \n\
The Linearity of this graph proves exponential convergence")
plt.semilogy(Delta_inf)
plt.xlabel("Iterate")
plt.ylabel(r'$|| V_{n+1} - V_n ||_\infty$')
plt.savefig('question6.png')
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
    """
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
    V_opt = np.zeros(env.nS)
    
    P = np.zeros((env.nA, env.nS, env.nS)) # Transition matrix, see slide 37 of the lecture "ShortLecture_MDPControl"
    m = np.zeros((env.nA, env.nS))
    for state, state_possiblities in env.P.items():
        for action in range(env.nA):
            for possibility in state_possiblities[action]: 
                proba, next_state, reward, done = possibility
                m[action, state] += reward * proba
                P[action, state, next_state] += proba

    delta_inf = []
    for _ in range(max_iter):
        V_old = V_opt
        V_opt = np.max(m + GAMMA * P @ V_old, axis=0)
        diff = np.max(np.abs(V_old - V_opt))
        delta_inf.append(diff)
        if diff < epsilon:
            break
    
    return V_opt, np.array(delta_inf)


starting_time = perf_counter()
V_opt, Delta_inf = value_function_optimal(1e-4, 10000)
print(convert_time(starting_time, perf_counter()))
print(f"Optimal value function:\n")
print_values(V_opt)

plt.figure()
plt.title("Semi-log graph of $n \mapsto || V_{n+1} - V_n ||_\infty $ \n\
The Linearity of this graph proves exponential convergence")
plt.semilogy(Delta_inf)
plt.xlabel("Iterate")
plt.ylabel(r"$|| V_{n+1} - V_n ||_\infty$")
plt.savefig('question7.png')
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
    pi : int array, size (env.nS,)
    An optimal policy
    """
    V_opt = np.zeros(env.nS)
    pi_opt = np.zeros(env.nS)
    
    P = np.zeros((env.nA, env.nS, env.nS)) # Transition matrix, see slide 37 of the lecture "ShortLecture_MDPControl"
    m = np.zeros((env.nA, env.nS))
    for state, state_possiblities in env.P.items():
        for action in range(env.nA):
            for possibility in state_possiblities[action]: 
                proba, next_state, reward, done = possibility
                m[action, state] += reward * proba
                P[action, state, next_state] += proba

    delta_inf = []
    for _ in range(max_iter):
        V_old = V_opt
        V_opt = np.max(m + GAMMA * P @ V_old, axis=0)
        pi_opt = np.argmax(m + GAMMA * P @ V_old, axis=0)
        diff = np.max(np.abs(V_old - V_opt))
        delta_inf.append(diff)
        if diff < epsilon:
            break
    
    return pi_opt


ARROWS = {
    fl.RIGHT: "→",
    fl.LEFT: "←",
    fl.UP: "↑",
    fl.DOWN: "↓"
}


def print_policy(pi):
    for row in range(env.nrow):
        for col in range(env.ncol):
            print(ARROWS[pi[to_s(row, col)]], end='')
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


def policy_improvement(v, cache=[None]):
    """
    V : float array, size (env.nS,)
    Value function of a policy

    returns:
    -------
    pi : int array, size (env.nS,)
    A policy that is greedy with respect to V
    """
    if cache[0] is None:
        P = np.zeros((env.nA, env.nS, env.nS)) # Transition matrix, see slide 37 of the lecture "ShortLecture_MDPControl"
        m = np.zeros((env.nA, env.nS))
        for state, state_possiblities in env.P.items():
            for action in range(env.nA):
                for possibility in state_possiblities[action]: 
                    proba, next_state, reward, done = possibility
                    m[action, state] += reward * proba
                    P[action, state, next_state] += proba
        data = (P, m)
        cache[0] =  data
    else :
        P, m = cache[0]
        
    pi = np.argmax(m + GAMMA * P @ v, axis=0)
    return pi


def policy_iteration(epsilon, max_iter):
    """
    epsilon : float
    Used as a threshold for the stopping rule

    max_iter : int
    Hard threshold on the number of loops

    returns:
    -------
    pi : int array, size (env.nS,)
    An optimal policy
    """
    V_opt = np.zeros(env.nS)
    pi_opt = np.zeros(env.nS, dtype=int)
    
    for _ in range(max_iter):
        pi_old = pi_opt
        # ----------------------------- policy evaluation ---------------------------- #
        V_opt, _ = value_function_2(pi_opt, epsilon, max_iter)
        # ---------------------------- policy improvement ---------------------------- #
        pi_opt = policy_improvement(V_opt)
        
        if np.all(pi_old==pi_opt):
            break
    return pi_opt


starting_time = perf_counter()
Pi_opt = policy_iteration(1e-40, 1000)
print(convert_time(starting_time, perf_counter()))
print("An optimal policy is:\n")
print_policy(Pi_opt)

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
    P = np.zeros((env.nA, env.nS, env.nS)) # Transition matrix, see slide 37 of the lecture "ShortLecture_MDPControl"
    m = np.zeros((env.nA, env.nS))
    for state, state_possiblities in env.P.items():
        for action in range(env.nA):
            for possibility in state_possiblities[action]: 
                proba, next_state, reward, done = possibility
                m[action, state] += reward * proba
                P[action, state, next_state] += proba
    
    q_opt = np.zeros((env.nA, env.nS))
    delta_inf = []
    for _ in range(max_iter):
        q_old = q_opt
        # ------------------------- Optimal Bellman operator ------------------------- #
        q_opt = m + GAMMA * P @ np.max(q_opt, axis=0)
        # --------------------------- difference evaluation -------------------------- #
        diff = np.max(np.abs(q_old - q_opt))
        delta_inf.append(diff)
        if diff < epsilon:
            break
    
    return q_opt.T, np.array(delta_inf)


starting_time = perf_counter()
Q_opt, Delta_inf = state_value_function_optimal(1e-4, 100)
print(convert_time(starting_time, perf_counter()))
# print(Q_opt)
V_opt = np.max(Q_opt, axis=1) 
print(f"Optimal value function:\n")
print_values(V_opt)

plt.figure()
plt.title("Semi-log graph of $n \mapsto || Q_{n+1} - Q_n ||_\infty $ \n\
The Linearity of this graph proves exponential convergence")
plt.semilogy(Delta_inf)
plt.xlabel("Iterate")
plt.ylabel(r"$|| Q_{n+1} - Q_n ||_\infty$")
plt.savefig('question11.png')
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
    cumulative_reward = 0
    discount = 1
    while not done and i < max_moves:
        i += 1
        _, r, done, _ = env.step(pi[env.s])
        cumulative_reward += discount*r
        discount *= GAMMA
        env.render()
        print('')
    return cumulative_reward


cr = trajectory(Pi_opt)
print("\nThe GOAL has been reached! Congrats! :-)")
print(f"The cumulative discounted reward along the above trajectory is: {cr:.3f}\n")


# %%
