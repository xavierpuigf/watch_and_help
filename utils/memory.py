from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import random
import pickle
from collections import deque, namedtuple
import pdb

Transition = namedtuple('Transition',
            ('state', 'policy', 'action', 'reward', 'mask'))


class MemoryMask():
    def __init__(self, max_episodes):
        self.num_episodes = max_episodes
        self.memory = deque(maxlen=self.num_episodes)
        self.memory.append([])
        self.position = 0
        self.max_reward = [-10]
        self.c_reward = [0]
        self.episode_counts = 0


    def reset(self):
        """reset memory"""
        self.memory = deque(maxlen=self.num_episodes)
        self.memory.append([])
        self.position = 0
        self.max_reward = [-10]
        self.c_reward = [0]
        self.episode_counts = 0


    def save(self, path):
        """save memory"""
        pickle.dump([[list(tran)] for episode in self.memory
                                     if len(episode) > 0
                                       for tran in episode],
                    open(path, 'wb'))


    def load(self, path):
        """load memory"""
        trajs = pickle.load(open(path))
        for traj in trajs:
            for tran in traj:
                self.append(*tran)


    def append(self, state, policy, action, reward, mask):
        """add new transition"""
        self.memory[self.position].append(Transition(state, policy, action, reward, mask))
        if reward > self.max_reward[self.position]:
          self.max_reward[self.position] = reward
        if reward is not None:
          self.c_reward[self.position] += reward
        # terminal states are saved with actions as None, so switch to next episode
        if action is None:
          if self.position + 1 >= self.num_episodes:
            self.memory.popleft()
            self.max_reward = self.max_reward[1:]
            self.c_reward = self.c_reward[1:]
          else:
            self.position = self.position + 1
            self.episode_counts += 1
          self.memory.append([])
          self.max_reward.append(-10)
          self.c_reward.append(0)


    def sample(self, maxlen=0):
        """samples random trajectory"""
        while True:
          e = random.randrange(len(self.memory))
          mem = self.memory[e]
          T = len(mem)
          if T > 0:
            if maxlen > 0 and T > maxlen + 1:
              t = random.randrange(T - maxlen - 1)
              return mem[t:t + maxlen + 1]
            else:
              return mem


    def sample_pos(self, maxlen=0):
        """sample positive experience"""
        while True:
          e = random.choice(self.list_pos)
          # if self.max_reward[e] < 0:
          # if self.c_reward[e] < 0:
          #   continue
          mem = self.memory[e]
          T = len(mem)
          if T > 0:
            if maxlen > 0 and T > maxlen + 1:
              t = random.randrange(T - maxlen - 1)
              return mem[t:t + maxlen + 1]
            else:
              return mem


    def sample_neg(self, maxlen=0):
        """sample negative experience"""
        while True:
          e = random.choice(self.list_neg)
          mem = self.memory[e]
          T = len(mem)
          if T > 0:
            if maxlen > 0 and T > maxlen + 1:
              t = random.randrange(T - maxlen - 1)
              return mem[t:t + maxlen + 1]
            else:
              return mem


    def sample_batch(self, batch_size, maxlen=0):
        """sample a batch"""
        if batch_size > self.episode_counts:
          batch = [self.sample(maxlen=maxlen) for _ in range(batch_size)]
          return list(map(list, zip(*batch)))
        idx = range(self.episode_counts)
        batch_idx = random.sample(idx, batch_size)
        batch = []
        for mem_idx in batch_idx:
          mem = self.memory[mem_idx]
          T = len(mem)
          if maxlen > 0 and T > maxlen + 1:
            t = random.randrange(T - maxlen - 1)
            batch.append(mem[t : t + maxlen + 1])
          else:
            batch.append(mem)
        return list(map(list, zip(*batch)))


    def sample_batch_balanced(self, batch_size, neg_ratio, maxlen = 0):
        """balanced batch sampling: pos vs neg"""
        N_pos, N_neg = 0, 0
        self.list_pos, self.list_neg = [], []
        for e in range(self.episode_counts):
          # if self.max_reward[e] > 0:
          if self.c_reward[e] > 0.0:
            N_pos += 1
            self.list_pos.append(e)
          elif len(self.memory[e]) > 0:
            N_neg += 1
            self.list_neg.append(e)
        print("pos:", N_pos)
        print("neg:", N_neg)
        if N_pos * N_neg == 0:
          return self.sample_batch(batch_size, maxlen)

        neg_batch_size = int(batch_size * neg_ratio)
        pos_batch_size = batch_size - neg_batch_size

        batch = [self.sample_pos(maxlen=maxlen) for _ in range(pos_batch_size)] \
              + [self.sample_neg(maxlen=maxlen) for _ in range(neg_batch_size)]

        return list(map(list, zip(*batch)))


    def __len__(self):
        return sum(len(episode) for episode in self.memory)