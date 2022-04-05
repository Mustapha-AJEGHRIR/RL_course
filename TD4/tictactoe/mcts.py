import abc
from collections import defaultdict


class MCTS(abc.ABC):
    """Monte Carlo Tree Search.
    Represent the game as a tree where the root is the current
    envirionment and the children are the environments obtained from the
    parent environment by playing a single action.
    For example at the start of a Tic Tac Toe game, the root is
    # # #
    # # #
    # # #
    and its has 9 children
    O # #   # O #   # # O   # # #   # # #   # # #   # # #   # # #   # # #
    # # #   # # #   # # #   O # #   # O #   # # O   # # #   # # #   # # #
    # # #   # # #   # # #   # # #   # # #   # # #   O # #   # O #   # # O
    each of which has 8 children and so on...

    If the first player chooses to play 1, the root becomes
    O # #
    # # #
    # # #
    and the first children are now
    O X #   O # X   O # #   O # #   O # #   O # #   O # #   O # #
    # # #   # # #   X # #   # X #   # # X   # # #   # # #   # # #
    # # #   # # #   # # #   # # #   # # #   X # #   # X #   # # X
    etc...

    The MCTS algoritm is composed of 4 training steps:
    1. Selection: explore the game tree from its current root until you find
        an explored descendent node (leaf).
    2. Expansion: discover the children of the selected leaf.
    3. Simulation: randomly play an action sequence starting from the leaf
        until the game stops.
    4. Backpropagation: update number of time played and cumulative rewards
        for each node visited on the path from root to terminal node in the
        simulation.

    At test time, the action recommended by MCTS is the one corresponding to
    the child of the root with highest average reward.
    """
    def __init__(self, player=1):
        # Wins (defaultdict(int) initialises unseen entries to 0)
        self.W = defaultdict(int)
        # Playouts (defaultdict(int) initialises unseen entries to 0)
        self.N = defaultdict(int)
        # Store children environment, obtained from the current one
        # by playing successive moves
        self.children = defaultdict(set)
        # Index of the player
        self.player = player

    def choose(self, node):
        """Choose the successor of node with highest average reward.
        """
        if node.is_over():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child().last_action

        def score(n):
            if self.N[n] == 0:
                return float('-inf')  # avoid unseen moves
            return self.W[n] / self.N[n]  # average reward

        best_successor = max(self.children[node], key=score)
        action = best_successor.last_action
        return action

    def playout(self, node):
        """Perform a single playout (one iteration of the MCTS training).
        """
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        """Find an unexplored descendent of node.
        """
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            # Descend one layer deeper
            node = self.search_heuristic(node)

    def _expand(self, node):
        """Discover children of node and keep track of them in the
        children dictionary.
        """
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    def _simulate(self, node):
        """Returns the reward for a random simulation starting from node and
        following the uniformly random policy until the game ends.
        """
        while True:
            if node.is_over():
                reward = node.reward(self.player)
                return reward
            node = node.find_random_child()

    def _backpropagate(self, path, reward):
        """Send the reward back up to the ancestors of the leaf.
        """
        for node in reversed(path):
            self.N[node] += 1
            self.W[node] += reward

    @abc.abstractmethod
    def search_heuristic(self, node):
        """Search heuristic to explore the game tree.
        """
        pass
