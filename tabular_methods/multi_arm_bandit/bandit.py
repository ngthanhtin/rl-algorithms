import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import random

class Bandit:
  def __init__(self, k=10, eps=0.2, lr=0.1, ucb=False, soft_max=False, c=2):
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
    self.initial_values = [] #optimistic initial value of each arm
    for i in range(self.k):
      self.initial_values.append(np.random.randn() + 1) #normal distribution
    #for ucb
    self.ucb = ucb
    self.times = 0
    self.c = c
    #for softmax action selection
    self.soft_max = soft_max

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
    #update avg reward using incremental formula
    self.record[action, 1] += self.lr*(r-self.record[action, 1])
    #update avg reward using original fomular
    # new_avg_reward = (self.record[action, 0] * self.record[action, 1] + r) / (self.record[action, 0]+1)
    # self.record[action, 1] = new_avg_reward
    #update observations
    self.record[action, 0] += 1
  
  def softmax(self, av, tau=1.12):
    softm = np.exp(av/tau)/np.sum(np.exp(av/tau))
    return softm

  def choose_action(self):
    if self.soft_max:
      p = self.softmax(self.record[:, 1], tau=0.7)
      action = np.random.choice(np.arange(self.k), p=p)  
      return action

    #explore
    if random.random() > self.eps:
      action=np.random.randint(self.k)
    #exploit
    else:
      if self.ucb:
        if self.times == 0:
          action = np.random.randint(self.k)
        else:
          confidence_bound = self.record[:, 1] + self.c*np.sqrt(np.log(self.times)/(self.times+0.1))
          action = np.argmax(confidence_bound)
      else:
        action=np.argmax(self.record[:, 1], axis=0)

    return action

  def play(self):
    rewards = [0]
    probs = np.random.rand(self.k) # random reward probabilities of each arm
    for i in range(500):
      action = self.choose_action()
      self.times += 1 #update for ucb
      r = self.get_reward(probs[action])
      r += self.initial_values[action] #optimistic initial value
      self.update_record(action, r)

      mean_reward = ((i+1)*rewards[-1] + r)/(i+2)
      rewards.append(mean_reward)

    fig,ax = plt.subplots(1,1)
    ax.set_xlabel("Plays")
    ax.set_ylabel("Avg Reward")
    fig.set_size_inches(9,5)
    ax.scatter(np.arange(len(rewards)), rewards)
    plt.show()

bandit = Bandit(k=10, soft_max=True)
bandit.play()