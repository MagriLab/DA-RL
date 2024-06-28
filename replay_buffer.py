from collections import deque
import random
import numpy as np

class ReplayBuffer():
  """A simple Python replay buffer."""

  def __init__(self, capacity):
    self.state = None
    self.action = None
    self.next_state = None
    self.buffer = deque(maxlen=capacity)

  def push(self, state, action, reward, terminated):
    self.state = self.next_state
    self.action = action
    self.next_state = state
    self.reward = reward
    self.terminated = terminated
    
    if action is not None: # so we keep the initial state but do not append it to the buffer
      self.buffer.append(
          (self.state, self.action, self.next_state, self.reward, self.terminated))

  def sample(self, batch_size):
    state, action, next_state, reward, terminated = zip(*random.sample(self.buffer, batch_size))
    return np.stack(state), np.stack(action), np.stack(next_state),  np.asarray(reward), np.asarray(terminated)

  def is_ready(self, batch_size):
    return batch_size <= len(self.buffer)