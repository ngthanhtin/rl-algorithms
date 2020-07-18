import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import random

class GradientBandit:
  def __init__(self, k=10, lr=0.1):
    """
    k: the number of bandits
    lr: step size in the incremental formula
    """
    self.k = k
    self.lr = lr
    self.initial_values = [] #optimistic initial value of each arm
    for i in range(self.k):
      self.initial_values.append(np.random.randn() + 1) #normal distribution

    #columns: numerical preference, softmax probability, observation and avg reward
    self.record = np.zeros((self.k, 4))
    #init softmax probability records
    self.record[:, 1] = 1/self.k

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
    #update numerical preference
    self.all_avg_reward = np.average(self.record[:, 3])
    self.record[action, 0] += self.lr*(r-self.all_avg_reward)*(1-self.record[action, 1])
    for other_action in range(self.k):
        if other_action != action:
            self.record[other_action, 0] = \
            self.record[other_action, 0] - self.lr*(r-self.all_avg_reward)*self.record[other_action, 1]
    # update softmax probability
    self.record[:, 1] = [self.softmax(a) for a in range(self.k)]
    #update avg reward using incremental formula
    self.record[action, 3] += self.lr*(r-self.record[action, 3])
    #update avg reward using original fomular
    # new_avg_reward = (self.record[action, 2] * self.record[action, 3] + r) / (self.record[action, 2]+1)
    # self.record[action, 3] = new_avg_reward
    #update observations
    self.record[action, 2] += 1
  
  def softmax(self, action):
    softm = np.exp(self.record[action, 0])/np.sum(np.exp(self.record[:, 0]))
    return softm

  def choose_action(self):
    action = np.random.choice(np.arange(self.k), p=self.record[:, 1])  
    return action

  def play(self):
    rewards = [0]
    probs = np.random.rand(self.k) # random reward probabilities of each arm
    for i in range(500):
      action = self.choose_action()
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

bandit = GradientBandit(k=10)
bandit.play()