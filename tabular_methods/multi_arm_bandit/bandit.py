import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import random

class Bandit:
  def __init__(self, k=10, eps=0.2, lr=0.1, ucb=False, c=2):
    """
    k: the number of bandits
    eps: e-greedy parameter
    lr: step size in the incremental formula
    ucb: upper confident bound
    c: a parameter of ucb
    """
    self.k = k
    self.eps = eps
    self.lr = lr
    self.ucb = ucb
    self.c = c
    #columns: Observation and avg reward
    self.record = np.zeros((self.k, 2))
  
  def get_reward(self, prob, n=10):
    """
    define reward distribution for an arm
    prob: reward probability
    n: the maximum reward will be 10
    """
    reward = 0
    for i in range(n):
      if random.random() < prob:
        reward += 1
    return reward
  
  def update_record(self, action, r):
    #update avg reward
    new_r = (self.record[action, 0] * self.record[action, 1] + r) / (self.record[action, 0]+1)
    self.record[action, 1] = new_r
    #update observations
    self.record[action, 0] += 1
    
  def get_best_arm(self):
    #choose action
    arm_index = np.argmax(self.record[:, 1], axis=0)
    return arm_index
  
  def choose_action(self):
    #exploit
    if random.random() < self.eps:
      action=self.get_best_arm()
    #explore
    else:
      action=np.random.randint(self.k)

    return action
  def play(self):
    rewards = [0]
    probs = np.random.rand(self.k) # random reward probabilities of each arm
    for i in range(500):
      if random.random() < self.eps:
        action = self.get_best_arm()
      else:
        action = np.random.randint(self.k)
      r = self.get_reward(probs[action])
      self.update_record(action, r)
      mean_reward = ((i+1)*rewards[-1] + r)/(i+2)
      rewards.append(mean_reward)

    fig,ax = plt.subplots(1,1)
    ax.set_xlabel("Plays")
    ax.set_ylabel("Avg Reward")
    fig.set_size_inches(9,5)
    ax.scatter(np.arange(len(rewards)), rewards)


bandit = Bandit(k=10)
bandit.play()