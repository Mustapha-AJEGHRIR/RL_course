import numpy as np
import gym
from copy import deepcopy


NBR_POS = 9
O_REWARD = 1
X_REWARD = -1
DRAW = 0
PRINT_DICT = {
        0: '#',
        1: 'O',
        2: 'X'
        }


class TicTacToeActions(gym.spaces.space.Space):
    """Dynamic action space for TicTacToe
    (action can be removed as they are played).
    """
    def __init__(self, n):
        assert n >= 2
        self.n = n
        self.reset()
        super(TicTacToeActions, self).__init__((), np.int64)

    def reset(self):
        self.available_actions = set([i + 1 for i in range(self.n)])

    def sample(self):
        return self.np_random.choice(list(self.available_actions))

    def remove_action(self, action):
        self.available_actions.remove(action)

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.char in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False
        return as_int in self.available_actions

    def __repr__(self):
        name = 'TicTacToeAction(%d;' % self.n
        for action in self.available_actions:
            name += ' %d' % action
        name += ')'
        return name

    def __eq__(self, other):
        return isinstance(other, TicTacToeActions) and self.available_actions == other.available_actions


class Board():
    """State of the TicTacToe game.
    """
    def __init__(self, board_array, board_map=PRINT_DICT, nbr_positions=NBR_POS):
        assert len(board_array) == nbr_positions, "Wrong board size"
        assert nbr_positions == int(np.sqrt(nbr_positions))**2, "Board is not square"
        self.board = board_array
        self.board_map = board_map
        self.nbr_positions = nbr_positions
        self.size = int(np.sqrt(nbr_positions))

    def __hash__(self):
        return hash(str(self.board))

    def __str__(self):
        s = ''
        for i, e in enumerate(self.board):
            assert e in self.board_map.keys(), f"Board element '{e}' is not a key of the PRINT_DICT"
            s += f"{self.board_map[e]} "
            if (i+1) % self.size == 0:
                if (i+1) != self.nbr_positions:
                    s += "\n"
        return s

    def board2idx(self):
        """Enumerate each board configuration by seeing
        it as a 9 digits number in base 3.
        (each board can be represented as
        [x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9]
        with x_i = 0, 1, 2,
        leading to 3 ** 9 configurations.)
        Basically a manual hash.
        """
        return int(np.dot(self.board, [3 ** n for n in range(self.nbr_positions)]))

    def board_and_keys(self):
        s = ""
        k = ""
        for i, e in enumerate(self.board):
            assert e in self.board_map.keys(), f"Board element '{e}' is not a key of the PRINT_DICT"
            s += f"{self.board_map[e]} "
            k += f"{i+1} "
            if (i+1) % self.size == 0:
                if (i+1) != self.nbr_positions:
                    s += f"   |   {k}"
                    s += "\n"
                    k = ""
        s += f"   |   {k}"
        return s

    def board_and_available_keys(self):
        s = ""
        k = ""
        for i, e in enumerate(self.board):
            assert e in self.board_map.keys(), f"Board element '{e}' is not a key of the PRINT_DICT"
            s += f"{self.board_map[e]} "
            k += f"{i+1 if e==0 else '-'} "
            if (i+1) % self.size == 0:
                if (i+1) != self.nbr_positions:
                    s += f"   |   {k}"
                    s += "\n"
                    k = ""
        s += f"   |   {k}"
        return s

    def play(self, pos, player):
        assert self.board[pos-1] == 0, "You cannot play here"
        self.board[pos-1] = player

    def check_state(self):
        for player in [1, 2]:
            test_arr = np.array([player for _ in range(self.size)])
            for i in range(self.size):
                if (self.board[(i*self.size):((i+1)*self.size)] == test_arr).sum() == self.size:
                    return player
                idx = [j*self.size+i for j in range(self.size)]
                if (self.board[idx] == test_arr).sum() == self.size:
                    return player
            idx = [i*(self.size+1) for i in range(self.size)]
            if (self.board[idx] == test_arr).sum() == self.size:
                return player
            idx = [(i+1)*(self.size-1) for i in range(self.size)]
            if (self.board[idx] == test_arr).sum() == self.size:
                return player
        if 0 in self.board:
            return -1
        return 0


class TicTacToeEnv(gym.Env):
    """Gym environment to play TicTacToe.
    """
    def __init__(self, nbr_pos=NBR_POS, force_tree=True):
        self.nbr_pos = nbr_pos
        self.board = Board(np.array([0 for _ in range(nbr_pos)]), PRINT_DICT, nbr_pos)
        self.current_player = 1
        self.action_space = TicTacToeActions(nbr_pos)
        self.pos_history = []
        # Technical: for MCTS, tree will be built as a dictionary,
        # whether same board states with different sequences of moves
        # are glued together (forming a DAG) or not depends on
        # the hash of TicTacToeEnv.
        self._force_tree = force_tree

    def __hash__(self):
        if self._force_tree:
            return hash(str(self) + str(self.pos_history))
        else:
            return hash(str(self))

    def reset(self):
        self.board = Board(np.array([0 for _ in range(self.nbr_pos)]), PRINT_DICT, self.nbr_pos)
        self.current_player = 1
        self.action_space.reset()
        self.pos_history = []
        board = deepcopy(self.board)
        return board

    def __str__(self):
        return self.board.board_and_keys()

    def state(self):
        return self.board.board

    @property
    def available_actions(self):
        return self.action_space.available_actions

    @property
    def last_action(self):
        return self.pos_history[-1]

    def sample(self):
        return self.action_space.sample()

    def reward(self, player):
        if self.is_over():
            if self.is_winner(player):
                reward = 1
            else:
                opponent = self.is_winner(player) % 2 + 1
                if self.is_winner(opponent):
                    reward = 0
                    # reward = -1
                else:
                    reward = 0.5  # DRAW
                    # reward = 0  # DRAW
        else:
            reward = 0
        return reward

    def step(self, pos, player=None):
        """The player whose turn it is plays at position pos.
        """
        assert self.action_space.contains(pos), 'Position {} is not available'.format(pos)
        self.pos_history.append(pos)

        if player is None:
            player = self.current_player
        self.board.play(pos, player)
        self.action_space.remove_action(pos)
        self.current_player = (player % 2) + 1

        reward = self.reward(player)
        done = self.is_over()

        new_board = deepcopy(self.board)
        return new_board, reward, done, {}

    def render(self, mode='human', close=False):
        print(self)
        print('\n')

    def is_winner(self, player):
        return self.board.check_state() == player

    def is_draw(self):
        return self.board.check_state() == 0

    def has_win(self):
        return self.board.check_state() == self.current_player

    def is_over(self):
        return not(self.board.check_state() == -1)

    def one_move_forward_env(self, pos):
        """Returns a copy of the environment after playing pos.
        """
        child_env = deepcopy(self)
        child_env.step(pos)
        return child_env

    def find_children(self):
        """Returns the set of all possible environments that are
        one action away from the current one.
        """
        if self.is_over():  # If the game is finished then no moves can be made
            return set()
        # Otherwise, you can make a move in each of the empty spots
        return {self.one_move_forward_env(pos) for pos in self.available_actions}

    def find_random_child(self):
        """Returns a copy of the environment after playing a uniformly random
        position among the available ones.
        """
        if self.is_over():
            return None
        else:
            return self.one_move_forward_env(self.sample())
