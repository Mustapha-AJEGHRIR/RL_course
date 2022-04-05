import numpy as np
import abc
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm
from tictactoe_env import TicTacToeEnv, DRAW


def play(ai_agent, game_env=TicTacToeEnv(), AI=2, params={}):
    print("**        Tic Tac Toe Game        **", flush=True)
    print(f"You are player {(AI % 2) + 1} and you play the symbol {game_env.board.board_map[(AI%2)+1]}\n", flush=True)
    game = game_env
    opponent = ai_agent(game, player=AI, params=params)
    loop = True
    while loop:
        game.reset()
        current_board = game.board
        while not game.is_over():
            current_player = game.current_player

            if current_player == AI:
                action = opponent.play()
            else:
                print(f"\n{game.board.board_and_available_keys()}")
                action = input("Where do you want to play? ")
                if action == "switch":
                    AI = (AI % 2) + 1
                    opponent.player = AI
                    action = opponent.play()
                    print(f"You are player {(AI % 2) + 1} and you play the symbol {current_board.board_map[(AI % 2)+1]}")
                elif action.isnumeric():
                    while int(action) not in game.available_actions:
                        action = input("Where do you want to play? ")
                    action = int(action)
                elif action == "quit":
                    loop = False
                    break
                else:
                    print('Unkown action, quitting...')
                    loop = False
                    break

            assert action in game.available_actions
            current_board, reward, done, _ = game.step(action)
            opponent.env = game

            if game.is_over():
                winner = current_board.check_state()
                print(winner)
                print("********************")
                if winner == 0:
                    print("**      DRAW      **")
                elif winner == AI:
                    print("**     AI WINS    **")
                else:
                    print("**     YOU WIN    **")
                print("********************")
                print('\n\n')


class Agent(abc.ABC):
    """Base class for agent.
    Method play needs to be instanciated.
    """
    def __init__(self, env, player=1):
        self._env = env
        self.player = player

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, new_env):
        self._env = new_env

    @abc.abstractmethod
    def play(self):
        pass


def agent_vs_agent(game,
                   agent1_class,
                   agent2_class,
                   n_episodes=1,
                   params1={},
                   params2={},
                   throttle=1,
                   plot=False,
                   ):
    """
    game: TicTacToeEnv

    agent1_class: Agent class
    First AI agent.

    agent2_class: Agent class
    Second AI agent.

    n_episodes : int
    Number of episodes.

    throttle: int
    Throttle tqdm updates.

    plot: bool
    If True, plot stats instead of returning them.

    returns:
    -------
    stats: array, number of draw, win and loss for each player.
    """
    stats = np.zeros(n_episodes)

    n_draw = 0
    n_nought_wins = 0
    n_cross_wins = 0
    postfix = {
            'draw': 0.0,
            '0 wins': 0.0,
            'X wins': 0.0,
        }

    _ = game.reset()

    agent1 = agent1_class(game, player=1, params=params1)
    agent2 = agent2_class(game, player=2, params=params2)

    with tqdm(total=n_episodes, postfix=postfix) as pbar:
        for episode in range(n_episodes):
            done = False
            _ = game.reset()
            turn = 0

            while not done:
                if turn % 2 == 0:
                    action = agent1.play()
                else:
                    action = agent2.play()

                _, _, done, _ = game.step(action)

                turn += 1

                if done:
                    break

            result = game.board.check_state()
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
    if plot:
        plot_stats(stats)
    else:
        return stats


def plot_stats(stats):
    draws = stats == DRAW
    nought_wins = stats == 1
    cross_wins = stats == 2

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(np.cumsum(draws) / np.cumsum(np.ones(len(stats))), color='green', label='draw')
    ax.plot(np.cumsum(nought_wins) / np.cumsum(np.ones(len(stats))), color='blue', label='0')
    ax.plot(np.cumsum(cross_wins) / np.cumsum(np.ones(len(stats))), color='red', label='X')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    ax.set_xlabel('')
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.show()


def randargmax(b, **kw):
    """ a random tie-breaking argmax"""
    return np.argmax(np.random.random(b.shape) * (b == b.max()), **kw)
