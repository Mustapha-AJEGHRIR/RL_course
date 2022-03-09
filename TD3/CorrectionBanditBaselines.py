import numpy as np
from math import log,sqrt
from BanditTools import *


class FTL:
    """Follow The Leader (a.k.a. greedy strategy)"""
    def __init__(self,nbArms):
        self.nbArms = nbArms
        self.clear()

    def clear(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)

    def chooseArmToPlay(self):
        if (min(self.nbDraws)==0):
            return randmax(-self.nbDraws)
        else:
            return randmax(self.cumRewards/self.nbDraws)

    def receiveReward(self,arm,reward):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] +1

    def name(self):
        return "FTL"


class UE:
    """Uniform Exploration"""
    def __init__(self,nbArms):
        self.nbArms = nbArms
        self.clear()

    def clear(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)

    def chooseArmToPlay(self):
        return np.random.randint(0,self.nbArms)

    def receiveReward(self,arm,reward):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] +1

    def name(self):
        return "UE"


class UCB:
    """Upper Confidence Bound"""
    def __init__(self,nbArms,delta):
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

    def receiveReward(self,arm,reward):
        self.time = self.time + 1
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] + 1
        self.means[arm] = self.cumRewards[arm] /self.nbDraws[arm]


        self.indexes = [self.means[a] + sqrt(log(1/self.delta(self.time))/(2*self.nbDraws[a])) if self.nbDraws[a] > 0 else np.Inf for a in range(self.nbArms)]

    def name(self):
        return "UCB"


class IMED:
    """Indexed Minimum Empirical Divergence"""
    def __init__(self,nbArms,kullback):
        self.nbArms = nbArms
        self.kl = kullback
        self.clear()

    def clear(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.means = np.zeros(self.nbArms)
        self.maxMeans = 0
        self.indexes = np.zeros(self.nbArms)

    def chooseArmToPlay(self):
        return randmin(self.indexes)

    def receiveReward(self,arm,reward):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] + 1
        self.means[arm] = self.cumRewards[arm] /self.nbDraws[arm]
        self.maxMeans = max(self.means)

        self.indexes = [self.nbDraws[a]*self.kl(self.means[a],self.maxMeans)+log(self.nbDraws[a]) if self.nbDraws[a] > 0 else 0 for a in range(self.nbArms)]

    def name(self):
        return "IMED"


class TS:
    """Thomson Sampling"""
    def __init__(self,nbArms):
        self.nbArms = nbArms
        self.clear()

    def clear(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.theta = np.zeros(self.nbArms)

    def chooseArmToPlay(self):
        return randmax(self.theta)

    def receiveReward(self,arm,reward):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] + 1

        self.theta = [np.random.beta(max(self.cumRewards[a],0) + 1, max(self.nbDraws[a] - self.cumRewards[a],0) + 1) for a in range(self.nbArms)]

    def name(self):
        return "TS"

class BESA:
    """ Best Empirical Sampled Average (2 arms) """
    def __init__(self):
        self.nbArms = 2
        self.clear()

    def clear(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.rewards = [[] for a in range(self.nbArms)]
        self.sampleSize = 0
        self.sample = [[] for a in range(self.nbArms)]
        self.means = np.zeros(self.nbArms)

    def chooseArmToPlay(self):
        if self.sampleSize==0:
            return randmin(self.nbDraws)
        else:
            return randmax(self.means)

    def receiveReward(self,arm,reward):
        self.rewards[arm] = self.rewards[arm]+[reward]
        self.nbDraws[arm] = self.nbDraws[arm] + 1
        self.sampleSize = int(min(self.nbDraws))

        self.samples = [ np.random.choice(self.rewards[a], size=self.sampleSize, replace=False) if self.sampleSize>0 else 0 for a in range(self.nbArms)]

        self.means = [ sum(self.samples[a])/self.sampleSize  if self.sampleSize>0 else 0 for a in range(self.nbArms) ]

    def name(self):
        return "BESA"





class IMED4UB:
    """Indexed Minimum Empirical Divergence for Unimodal Bandits"""
    def __init__(self,nbArms,kullback):
        self.nbArms = nbArms
        self.kl = kullback
        self.clear()

    def clear(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.means = np.zeros(self.nbArms)
        self.maxMeans = 0
        self.bestArm = randmax(self.means)
        self.indexes = np.zeros(self.nbArms)
        self.structuredIndexes = np.zeros(self.nbArms)

    def chooseArmToPlay(self):
        return randmin(self.structuredIndexes)

    def receiveReward(self,arm,reward):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] + 1
        self.means[arm] = self.cumRewards[arm] /self.nbDraws[arm]
        self.bestArm = randmax(self.means)
        self.maxMeans = self.means[self.bestArm]

        self.indexes = [self.nbDraws[a]*self.kl(self.means[a],self.maxMeans)+log(self.nbDraws[a]) if self.nbDraws[a] > 0 else 0 for a in range(self.nbArms)]

        self.structuredIndexes = [self.indexes[a] if abs(a - self.bestArm) <=1 else np.Infinity for a in range(self.nbArms)]

    def name(self):
        return "IMED4UB"


class IMED4LB:
    """Indexed Minimum Empirical Divergence for Lipschitz Bandits"""
    def __init__(self, nbArms, kullback, LipschitzConstant):
        self.nbArms = nbArms
        self.kl = kullback
        self.L = LipschitzConstant
        self.clear()

    def clear(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.means = np.zeros(self.nbArms)
        self.maxMeans = 0
        self.bestArm = randmax(self.means)
        self.structuredIndexes = np.zeros(self.nbArms)

    def chooseArmToPlay(self):
        return randmin(self.structuredIndexes)

    def receiveReward(self,arm,reward):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] + 1
        self.means[arm] = self.cumRewards[arm] /self.nbDraws[arm]
        self.bestArm = randmax(self.means)
        self.maxMeans = self.means[self.bestArm]

        self.structuredIndexes = []
        for a in range(self.nbArms):
            if self.means[a]==self.maxMeans:
                if self.nbDraws[a] == 0:
                    self.structuredIndexes = self.structuredIndexes + [0]
                else:
                    self.structuredIndexes = self.structuredIndexes + [log(self.nbDraws[a])]
            else:
                i = 0
                for b in range(self.nbArms):
                    if self.nbDraws[b] > 0:
                        if self.means[b] < self.maxMeans - self.k*abs(a-b):
                            i = i + self.nbDraws[b]*self.kl(self.means[b],self.maxMeans - self.L*abs(a-b))+log(self.nbDraws[b])
                self.structuredIndexes = self.structuredIndexes + [i]

    def name(self):
        return "IMED4LB"


