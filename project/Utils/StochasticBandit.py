import statistics
import numpy as np
import Arms as arms
from BanditTools import Scheduler


class MAB_NON_S:
    """Non-stationary MAB"""
    def __init__(self, k=3,
                distribution= "bernoulli",
                seed = 42,
                TS = [1000], # T's for which distribution is changed. will loop over TS  periodically 
                ):
        self.historical_means = []
        self.historical_arms = []
        self.historical_rewards = []
        self.pseudoRegret = []
        self.distribution = distribution
        self.nbArms = k
        self.scheduler = Scheduler(TS)

        self.generate_distributions()
    def restore(self):
        """
        The function restores the scheduler and the historical data
        """
        self.scheduler.restore()
        self.historical_means = []
        self.historical_arms = []
        self.historical_rewards = []
        self.pseudoRegret = []
    def generate_distributions(self):
        """
        Generate the means randomly and the distributions
        """
        self.means = np.random.rand(self.nbArms)

        if self.distribution == "bernoulli":
            self.arms = [arms.Bernoulli(p) for p in self.means]
        elif self.distribution == "gaussian":
            self.arms = [arms.Gaussian(p) for p in self.means]
        else :
            raise ValueError("Distribution not supported") 
    def generateReward(self, arm):
        reward = self.arms[arm].sample()
        self.historical_arms(arm)
        self.historical_rewards(reward)
        self.historical_means(self.means)
        self.pseudoRegret.append(np.max(self.means) - self.means[arm])
        
        change = self.scheduler.next_bandits() # check wether we need to change the bandits
        if change:
            self.generate_distributions()
        
        return reward
    # ------------------------------ Use statistics ------------------------------ #
    def CumulativeRegret(self):
        """
        Compute the Cumulative pseudo Regret, not the regret ! 
        """
        return np.cumsum(self.pseudoRegret)



class MAB:
    def __init__(self,arms, distributionType = 'unknown', structureType = 'unknown', structureParameter = None):
        """given a list of arms, create the MAB environnement"""
        self.arms = arms
        self.nbArms = len(arms)
        self.means = [arm.mean for arm in arms]
        self.bestarm = np.argmax(self.means)
        self.distribution = distributionType
        self.structure = structureType
        self.parameter = structureParameter
    
    def generateReward(self,arm):
        return self.arms[arm].sample()

## some functions that create specific MABs

def BernoulliBandit(means, structure = 'unknown',parameter = None):
    """define a Bernoulli MAB from a vector of means"""
    return MAB([arms.Bernoulli(p) for p in means], distributionType = 'Bernoulli', structureType = structure, structureParameter = parameter)

def GaussianBandit(means,var=1, structure = 'unknown', parameter = None):
    """define a Gaussian MAB from a vector of means"""
    return MAB([arms.Gaussian(p,var) for p in means], distributionType = 'Gaussian', structureParameter = parameter)

def RandomBernoulliBandit(Delta,K):
    """generates a K-armed Bernoulli instance at random where Delta is the gap between the best and second best arm"""
    maxMean = Delta + np.random.rand()*(1.-Delta)
    secondmaxMean= maxMean-Delta
    means = secondmaxMean*np.random.random(K)
    bestarm = np.random.randint(0,K)
    secondbestarm = np.random.randint(0,K)
    while (secondbestarm==bestarm):
        secondbestarm = np.random.randint(0,K)
    means[bestarm]=maxMean
    means[secondbestarm]=secondmaxMean
    return BernoulliBandit(means)
