import numpy as np
from math import log, sqrt
from BanditTools import *


class FTL:
    """Follow The Leader (a.k.a. greedy strategy)"""

    def __init__(self, nbArms):
        self.nbArms = nbArms
        self.clear()

    def clear(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)

    def chooseArmToPlay(self):
        if (min(self.nbDraws) == 0):
            # Means, return one arm that have never been drawn before
            return randmax(-self.nbDraws)
        else:
            return randmax(self.cumRewards/self.nbDraws)

    def receiveReward(self, arm, reward):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] + 1

    def name(self):
        return "FTL"


class UE:
    """Uniform Exploration"""

    def __init__(self, nbArms):
        self.nbArms = nbArms
        self.clear()

    def clear(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)

    def chooseArmToPlay(self):
        return np.random.randint(0, self.nbArms)

    def receiveReward(self, arm, reward):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] + 1

    def name(self):
        return "UE"


class UCB:
    """Upper Confidence Bound"""

    def __init__(self, nbArms, delta):
        self.nbArms = nbArms
        self.delta = delta
        self.clear()

    def clear(self):
        self.time = 0
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.means = np.zeros(self.nbArms)
        self.indexes = np.zeros(self.nbArms)

    def chooseArmToPlay(self):
        return randmax(self.indexes)

    def receiveReward(self, arm, reward):
        self.time = self.time + 1
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] + 1
        self.means[arm] = self.cumRewards[arm] / self.nbDraws[arm]

        self.indexes = [self.means[a] + sqrt(log(1/self.delta(self.time))/(
            2*self.nbDraws[a])) if self.nbDraws[a] > 0 else np.Inf for a in range(self.nbArms)]

    def name(self):
        return "UCB"

# class UCB:
#     """Upper Confidence Bound"""

#     def __init__(self, nbArms, delta):
#         self.nbArms = nbArms
#         self.delta = delta
#         self.clear()

#     def clear(self):
#         self.time = 0
#         self.nbDraws = np.zeros(self.nbArms)
#         self.cumRewards = np.zeros(self.nbArms)
#         self.means = np.zeros(self.nbArms)
#         self.indexes = np.zeros(self.nbArms)

#     def chooseArmToPlay(self):
#         return randmax(self.indexes)

#     def receiveReward(self, arm, reward):
#         self.time = self.time + 1
#         self.cumRewards[arm] = self.cumRewards[arm]+reward
#         self.nbDraws[arm] = self.nbDraws[arm] + 1
#         self.means[arm] = self.cumRewards[arm] / self.nbDraws[arm]

#         self.indexes = [self.means[a] + sqrt(log(1/self.delta(self.time))/(
#             2*self.nbDraws[a])) if self.nbDraws[a] > 0 else np.Inf for a in range(self.nbArms)]

#     def name(self):
#         return "UCB"


class KL_UCB:
    """kullback leibler Upper Confidence Bound"""

    def __init__(self, nbArms, delta, klucb=klucbBern, c=1, tolerance=1e-4):
        self.nbArms = nbArms
        self.delta = delta
        self.klucb_vect = np.vectorize(klucb)
        self.c = c
        self.tolerance = tolerance
        self.clear()

    def clear(self):
        self.time = 0
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.means = np.zeros(self.nbArms)
        self.indexes = np.zeros(self.nbArms)

    def chooseArmToPlay(self):
        return randmax(self.indexes)

    def receiveReward(self, arm, reward):
        self.time = self.time + 1
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] + 1
        self.means[arm] = self.cumRewards[arm] / self.nbDraws[arm]

        self.indexes = self.klucb_vect(
            self.cumRewards / self.nbDraws, self.c * np.log(self.time) / self.nbDraws, self.tolerance)
        self.indexes[self.nbDraws < 1] = float('+inf')

    def name(self):
        return "KL_UCB"


class R_UCB:
    """Refreshed Upper Confidence Bound"""

    def __init__(self, nbArms, delta, period):
        self.nbArms = nbArms
        self.delta = delta
        self.period = period
        self.clear()

    def clear(self):
        self.time = 0
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.means = np.zeros(self.nbArms)
        self.indexes = np.zeros(self.nbArms)

    def chooseArmToPlay(self):
        return randmax(self.indexes)

    def receiveReward(self, arm, reward):
        if self.time == self.period:
            self.clear()
        self.time = self.time + 1
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] + 1
        self.means[arm] = self.cumRewards[arm] / self.nbDraws[arm]

        self.indexes = [self.means[a] + sqrt(log(1/self.delta(self.time))/(
            2*self.nbDraws[a])) if self.nbDraws[a] > 0 else np.Inf for a in range(self.nbArms)]

    def name(self):
        return "R_UCB"


class SW_UCB:
    """Sliding Window Upper Confidence Bound"""

    def __init__(self, nbArms, delta, buffer_size):
        self.nbArms = nbArms
        self.delta = delta
        self.buffer_size = buffer_size

        self.clear()

    def clear(self):
        self.time = 0
        self.draws = DrawsBuffer(self.nbArms, self.buffer_size)
        self.means = np.zeros(self.nbArms)
        self.indexes = np.zeros(self.nbArms)

    def chooseArmToPlay(self):
        return randmax(self.indexes)

    def receiveReward(self, arm, reward):
        self.time = min(self.time + 1, self.buffer_size)
        self.draws.add(arm, reward)
        self.cumRewards = self.draws.get_cum_rewards()
        self.nbDraws = self.draws.get_nb_draws()
        for arm in range(self.nbArms):
            self.means[arm] = self.cumRewards[arm] / self.nbDraws[arm]

        self.indexes = [self.means[a] + sqrt(log(1/self.delta(self.time))/(
            2*self.nbDraws[a])) if self.nbDraws[a] > 0 else np.Inf for a in range(self.nbArms)]

    def name(self):
        return "SW_UCB"


class IMED:
    """Index Minimum Empirical Divergence"""

    def __init__(self, nbArms, kullback):
        self.nbArms = nbArms
        self.kl = kullback
        self.clear()

    def clear(self):
        self.time = 0
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.means = np.zeros(self.nbArms)
        self.maxMeans = 0
        self.indexes = np.zeros(self.nbArms)

    def chooseArmToPlay(self):
        return randmin(self.indexes)

    def receiveReward(self, arm, reward):
        self.time = self.time + 1
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] + 1
        self.means[arm] = self.cumRewards[arm] / self.nbDraws[arm]
        self.maxMeans = max(self.means)

        self.indexes = [self.nbDraws[a]*self.kl(self.means[a], self.maxMeans) + log(
            self.nbDraws[a]) if self.nbDraws[a] > 0 else -np.Inf for a in range(self.nbArms)]

    def name(self):
        return "IMED"


class IMED4UB:
    """Index Minimum Empirical Divergence for Unimodal Bandits"""

    def __init__(self, nbArms, kullback):
        self.nbArms = nbArms
        self.kl = kullback
        self.clear()

    def clear(self):
        self.time = 0
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.means = np.zeros(self.nbArms)
        self.maxMeans = 0
        self.indexes = np.zeros(self.nbArms)
        self.structuredIndexes = np.zeros(self.nbArms)

    def chooseArmToPlay(self):
        return randmin(self.structuredIndexes)

    def receiveReward(self, arm, reward):
        self.time = self.time + 1
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] + 1
        self.means[arm] = self.cumRewards[arm] / self.nbDraws[arm]
        self.bestArm = randmax(self.means)
        self.maxMeans = self.means[self.bestArm]

        self.indexes = [self.nbDraws[a]*self.kl(self.means[a], self.maxMeans) + log(
            self.nbDraws[a]) if self.nbDraws[a] > 0 else -np.Inf for a in range(self.nbArms)]

        self.structuredIndexes = [self.indexes[a] if abs(
            a-self.bestArm) <= 1 else np.Inf for a in range(self.nbArms)]

    def name(self):
        return "IMED4UB"
