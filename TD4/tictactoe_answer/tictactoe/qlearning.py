import numpy as np
from tqdm import tqdm
from tictactoe_env import DRAW
from utils import Agent, randargmax


class QFunction():
    def __init__(self, nbr_pos=9):
        assert nbr_pos == int(np.sqrt(nbr_pos))**2, "Board is not square"
        self.nbr_pos = nbr_pos
        self.Q = np.zeros((3 ** nbr_pos, nbr_pos))
        self.idx2pos = {
            0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9
        }
        self.pos2idx = {
            1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8
        }

    def board2idx(self, board):
        """Enumerate each board configuration by seeing
        it as a 9 digits number in base 3.
        (each board can be represented as
        [x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9]
        with x_i = 0, 1, 2,
        leading to 3 ** 9 configurations.)
        Basically a manual hash.
        """
        return int(np.dot(board, [3 ** n for n in range(self.nbr_pos)]))

    def available_actions2idx(self, env):
        return [self.pos2idx[i] for i in env.available_actions]

    def update_Q(self, state, action, new_value):
        state_idx = self.board2idx(state)
        action_idx = self.pos2idx[action]
        self.Q[state_idx, action_idx] = new_value

    def get_Q(self, state, action):
        state_idx = self.board2idx(state)
        action_idx = self.pos2idx[action]
        return self.Q[state_idx, action_idx]

    def get_max_action_Q(self, state, env=None):
        state_idx = self.board2idx(state)
        if env is None:
            return np.max(self.Q[state_idx])
        else:
            available_actions_idx = self.available_actions2idx(env)
            if len(available_actions_idx) == 0:
                return 0.0
            else:
                return np.max(self.Q[state_idx, available_actions_idx])

    def get_argmax_action_Q(self, state, env=None):
        state_idx = self.board2idx(state)
        if env is None:
            return randargmax(self.Q[state_idx])
        else:
            available_actions_idx = self.available_actions2idx(env)
            return randargmax(self.Q[state_idx, available_actions_idx])

    def get_argmax_action_Q_idx(self, state, env):
        available_actions_idx = self.available_actions2idx(env)
        return self.idx2pos[
            available_actions_idx[
                self.get_argmax_action_Q(state, env)
                ]
            ]


class QLearningAgent(Agent):
    """Q-Learning TicTacToe agent.
    """
    def __init__(self, env, player=1, params={'Q': None}):
        self._env = env
        self.player = player  # index of the player
        Q = params.get('Q', None)
        if Q is not None:
            self.Q = Q
        else:
            self.Q = QFunction(nbr_pos=env.nbr_pos)

    def play(self):
        """Syntax: play should be a metbod without argument
        except self.
        """
        return self.Q.get_argmax_action_Q_idx(self.env.state(), self.env)


def Q_learning_vs_random(env,
                         n_episodes,
                         learning_rate,
                         epsilon_greedy,
                         epsilon_greedy_decay_freq,
                         epsilon_greedy_decay_factor,
                         gamma,
                         Q=None,
                         throttle=100,
                         ):
    """
    env: TicTacToeEnv

    n_episodes : int
    Number of episodes

    learning_rate : float
    Constant learning rate for the TD step.

    epsilon_greedy : float
    Initial probability of random exploration.

    epsilon_greedy_freq : int
    Decrease epsilon_greedy every epsilon_greedy_decay episodes.

    epsilon_greedy_factor : float
    Decrease epsilon_greedy by (1-epsilon_greedy_factor).

    Q: QFunction
    If not None, initial Q function for Q-learning agent.

    gamma: float
    Discount factor.

    throttle: int
    Throttle tqdm updates

    returns:
    -------
    Q : float array, (env.nS, env.nA) size
    Optimal state-action value function on the TicTacToa MDP
    Q[state index][action index] = state-action value of that state

    stats: array, number of draw, win and loss for each player.
    """
    agent = QLearningAgent(env, params={'Q': Q})

    stats = np.zeros(n_episodes)

    n_draw = 0
    n_nought_wins = 0
    n_cross_wins = 0
    postfix = {
            'draw': 0.0,
            '0 wins': 0.0,
            'X wins': 0.0,
        }
    with tqdm(total=n_episodes, postfix=postfix) as pbar:
        for episode in range(n_episodes):
            done = False
            state = env.reset()

            if (episode + 1) % epsilon_greedy_decay_freq == 0:
                epsilon_greedy *= epsilon_greedy_decay_factor
            while not done:
                # epsilon-greedy exploration
                if np.random.rand() < epsilon_greedy:
                    action = env.sample()
                else:
                    action = agent.play()

                next_state, r, done, _ = env.step(action)

                pred = agent.Q.get_Q(state.board, action)
                target = r + gamma * agent.Q.get_max_action_Q(next_state.board)
                agent.Q.update_Q(state.board, action, pred + learning_rate * (target - pred))

                # Move on to next state
                state = next_state

                if done:
                    break

                action = env.sample()

                next_state, r, done, _ = env.step(action)

                # Move on to next state
                state = next_state

            result = env.board.check_state()
            stats[episode] = result
            n_draw += int(result == DRAW)
            n_nought_wins += int(result == 1)
            n_cross_wins += int(result == 2)
            postfix['draw'] = '{:.0%}'.format(n_draw / (episode + 1))
            postfix['0 wins'] = '{:.0%}'.format(n_nought_wins / (episode + 1))
            postfix['X wins'] = '{:.0%}'.format(n_cross_wins / (episode + 1))

            if episode % throttle == 0:
                pbar.set_postfix(postfix)
                pbar.update(throttle)
    return agent, stats


def random_vs_Q_learning(env,
                         n_episodes,
                         learning_rate,
                         epsilon_greedy,
                         epsilon_greedy_decay_freq,
                         epsilon_greedy_decay_factor,
                         gamma,
                         Q=None,
                         throttle=100,
                         ):
    """
    env: TicTacToeEnv

    n_episodes : int
    Number of episodes

    learning_rate : float
    Constant learning rate for the TD step.

    epsilon_greedy : float
    Initial probability of random exploration.

    epsilon_greedy_freq : int
    Decrease epsilon_greedy every epsilon_greedy_decay episodes.

    epsilon_greedy_factor : float
    Decrease epsilon_greedy by (1-epsilon_greedy_factor).

    Q: QFunction
    If not None, initial Q function for Q-learning agent.

    gamma: float
    Discount factor.

    throttle: int
    Throttle tqdm updates

    returns:
    -------
    Q : float array, (env.nS, env.nA) size
    Optimal state-action value function on the TicTacToa MDP
    Q[state index][action index] = state-action value of that state

    stats: array, number of draw, win and loss for each player.
    """
    agent = QLearningAgent(env, params={'Q': Q})

    stats = np.zeros(n_episodes)

    n_draw = 0
    n_nought_wins = 0
    n_cross_wins = 0
    postfix = {
            'draw': 0.0,
            '0 wins': 0.0,
            'X wins': 0.0,
        }
    with tqdm(total=n_episodes, postfix=postfix) as pbar:
        for episode in range(n_episodes):
            done = False
            state = env.reset()

            if (episode + 1) % epsilon_greedy_decay_freq == 0:
                epsilon_greedy *= epsilon_greedy_decay_factor
            while not done:
                action = env.sample()

                next_state, r, done, _ = env.step(action)

                # Move on to next state
                state = next_state

                if done:
                    break

                # epsilon-greedy exploration
                if np.random.rand() < epsilon_greedy:
                    action = env.sample()
                else:
                    action = agent.play()

                next_state, r, done, _ = env.step(action)

                pred = agent.Q.get_Q(state.board, action)
                target = r + gamma * agent.Q.get_max_action_Q(next_state.board, env)
                agent.Q.update_Q(state.board, action, pred + learning_rate * (target - pred))

                # Move on to next state
                state = next_state

            result = env.board.check_state()
            stats[episode] = result
            n_draw += int(result == DRAW)
            n_nought_wins += int(result == 1)
            n_cross_wins += int(result == 2)
            postfix['draw'] = '{:.0%}'.format(n_draw / (episode + 1))
            postfix['0 wins'] = '{:.0%}'.format(n_nought_wins / (episode + 1))
            postfix['X wins'] = '{:.0%}'.format(n_cross_wins / (episode + 1))

            if episode % throttle == 0:
                pbar.set_postfix(postfix)
                pbar.update(throttle)
    return agent, stats
