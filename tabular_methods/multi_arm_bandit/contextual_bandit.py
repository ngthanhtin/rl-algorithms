import torch as th
import numpy as np
from torch.autograd import Variable
from matplotlib import pyplot as plt
import random

class ContextBandit:
  def __init__(self, arms=10):
    self.arms = arms
    self.init_distribution(arms)
    self.update_state()
  
  def init_distribution(self, arms):
    #num states equals num arms to keep things simple
    self.bandit_matrix = np.random.rand(arms, arms)
  
  def reward(self, prob, n=10):
    reward = 0
    for i in range(n):
      if random.random() < prob:
        reward += 1
    return reward
    
  def get_state(self):
    return self.state

  def update_state(self):
    self.state = np.random.randint(0, self.arms)

  def get_reward(self, arm):
    return self.reward(self.bandit_matrix[self.get_state()][arm])
  
  def choose_arm(self, arm):
    reward = self.get_reward(arm)
    self.update_state()
    return reward


def softmax(av, tau=0.7):
  n = len(av)
  probs = np.zeros(n)
  for i in range(n):
    softm = (np.exp(av[i]/tau)/np.sum(np.exp(av[:]/tau)))
    probs[i] = softm
  return probs

def one_hot(N, pos, val=1):
  one_hot_vec = np.zeros(N)
  one_hot_vec[pos] = val
  return one_hot_vec

arms = 10
# N is the batch size, D_in is input dimension
# H is hidden dimension D_out is output dimension
N, D_in, H, D_out = 1, arms, 100, arms

model = th.nn.Sequential(
    th.nn.Linear(D_in, H),
    th.nn.ReLU(),
    th.nn.Linear(H, D_out),
    th.nn.ReLU(),
)
loss_fn = th.nn.MSELoss(size_average=False)

env = ContextBandit(arms)

def train(env):
  epochs = 50000
  # one-hot encode current state
  cur_state = Variable(th.Tensor(one_hot(arms, env.get_state())))
  reward_hist = np.zeros(50)
  reward_hist[:] = 5
  runningMean = np.average(reward_hist)
  lr = 1e-2
  optimizer = th.optim.Adam(model.parameters(), lr=lr)
  plt.xlabel("Plays")
  plt.ylabel("Mean Reward")
  for i in range(epochs):
    y_pred = model(cur_state) # produce reward prediction
    av_softmax = softmax(y_pred.data.numpy(), tau=2.0) #turn reward distribution into probability distribution
    av_softmax /= av_softmax.sum() #make sure total prob adds to 1
    action = np.random.choice(arms, p=av_softmax)
    cur_reward = env.choose_arm(action)
    one_hot_reward = y_pred.data.numpy().copy()
    one_hot_reward[action] = cur_reward
    reward = Variable(th.Tensor(one_hot_reward))
    loss=loss_fn(y_pred, reward)

    if i%50 == 0:
      runningMean = np.average(reward_hist)
      reward_hist[:] = 0
      plt.scatter(i, runningMean)
    reward_hist[i%50]=cur_reward
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
    cur_state = Variable(th.Tensor(one_hot(arms, env.get_state())))

train(env)