# -*- coding: utf-8 -*-
'''
Useful functions for bandit algorithms (especially KL-UCB)
'''

from math import log, sqrt, exp
import numpy as np

class DrawsBuffer:
    def __init__(self, nbArms, size):
        self.size = size
        self.nbArms = nbArms
        self.buffer_arm = np.zeros(size, dtype=np.int32) - 1 # -1 means not initialized
        self.buffer_reward = np.zeros(size)

        self.draws = np.zeros(nbArms, dtype=np.int32)
        self.cumRewards = np.zeros(nbArms)
        self.index = 0
    def get_nb_draws(self):
        return self.draws
    def get_cum_rewards(self):
        return self.cumRewards
    def old(self):
        return self.buffer_arm[self.index], self.buffer_reward[self.index]
    def add(self, arm, reward):
        old_arm, old_reward = self.old()
        if old_arm == -1:
            self.buffer_arm[self.index] = arm
            self.buffer_reward[self.index] = reward
        else:
            self.buffer_arm[self.index] = arm
            self.buffer_reward[self.index] = reward
            self.draws[old_arm] -= 1
            self.cumRewards[old_arm] -= old_reward
        self.draws[arm] += 1
        self.cumRewards[arm] += reward
        self.index = (self.index + 1) % self.size
    def __len__(self):
        return self.nbArms
    def __getitem__(self, key):
        return self.draws[key], self.cumRewards[key]
    def __str__(self):
        return "Draws = " + str(self.draws) + "\nCumRewards = "+  str(self.cumRewards)
        


class Scheduler:
    """
    cheks if it is time change the distibution of bandits for the Non stationary bandits
    """
    def __init__(self, list):
        self.list = list
        self.index = 0
        self.time = 0
    def restore(self):
        self.index = 0
        self.time = 0
    def get(self):
        return self.list[self.index] 
    def next_bandits(self):
        self.time += 1
        T = self.get()
        if self.time == T:
            self.index = (self.index + 1) % len(self.list)
            self.time = 0
            return True
        else:
            return False


## A function that returns an argmax at random in case of multiple maximizers 

def randmax(A):
    maxValue=max(A)
    index = [i for i in range(len(A)) if A[i]==maxValue]
    return np.random.choice(index)

## A function that returns an argmin at random in case of multiple maximizers 

def randmin(A):
    minValue=min(A)
    index = [i for i in range(len(A)) if A[i]==minValue]
    return np.random.choice(index)

## Kullback-Leibler divergence in exponential families 

eps = 1e-15

def klBern(x, y):
    """Kullback-Leibler divergence for Bernoulli distributions."""
    x = min(max(x, eps), 1-eps)
    y = min(max(y, eps), 1-eps)
    return x*log(x/y) + (1-x)*log((1-x)/(1-y))


def klGauss(x, y, sig2 = 1.):
    """Kullback-Leibler divergence for Gaussian distributions."""
    return (x-y)*(x-y)/(2*sig2)


def klPoisson(x, y):
    """Kullback-Leibler divergence for Poison distributions."""
    x = max(x, eps)
    y = max(y, eps)
    return y-x+x*log(x/y)


def klExp(x, y):
    """Kullback-Leibler divergence for Exponential distributions."""
    x = max(x, eps)
    y = max(y, eps)
    return (x/y - 1 - log(x/y))


## computing the KL-UCB indices 

def klucb(x, level, div, upperbound, lowerbound=-float('inf'), precision=1e-6):
    """Generic klUCB index computation using binary search: 
    returns u>x such that div(x,u)=level where div is the KL divergence to be used.
    """
    l = max(x, lowerbound)
    u = upperbound
    while u-l>precision:
        m = (l+u)/2
        if div(x, m)>level:
            u = m
        else:
            l = m
    return (l+u)/2


def klucbBern(x, level, precision=1e-6):
    """returns u such that kl(x,u)=level for the Bernoulli kl-divergence."""
    upperbound = min(1.,x+sqrt(level/2)) 
    return klucb(x, level, klBern, upperbound, precision)


def klucbGauss(x, level, sig2=1., precision=0.):
    """returns u such that kl(x,u)=level for the Gaussian kl-divergence (can be done in closed form).    
    """
    return x + sqrt(2*sig2*level)


def klucbPoisson(x, level, precision=1e-6):
    """returns u such that kl(x,u)=level for the Poisson kl-divergence."""
    upperbound = x+level+sqrt(level*level+2*x*level) 
    return klucb(x, level, klPoisson, upperbound, precision)


def klucbExp(x, d, precision=1e-6):
    """returns u such that kl(x,u)=d for the exponential kl divergence."""
    if d<0.77:
        upperbound = x/(1+2./3*d-sqrt(4./9*d*d+2*d)) # safe, klexp(x,y) >= e^2/(2*(1-2e/3)) if x=y(1-e)
    else:
        upperbound = x*exp(d+1)
    if d>1.61:
        lowerbound = x*exp(d)
    else:
        lowerbound = x/(1+d-sqrt(d*d+2*d))
    return klucb(x, d, klExp, upperbound, lowerbound, precision)

# Computing the complexity of a bandit instance
def complexity(bandit):
    """ computes the complexity of a Bernoulli or Gaussian unstructured bandit instance """
    meanMax = max(bandit.means)
    
    if bandit.distribution == 'Gaussian':
        return sum([(meanMax - bandit.means[a])/klGauss(bandit.means[a],meanMax) for a in range(bandit.nbArms) if a!=bandit.bestarm])
    
    else: 
        return sum([(meanMax - bandit.means[a])/klBern(bandit.means[a],meanMax) for a in range(bandit.nbArms) if a!=bandit.bestarm])

        

    
        