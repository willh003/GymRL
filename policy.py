import random
import numpy as np
from collections import defaultdict


class Policy:
    """
    Interface for a policy
    """
    def __init__(self):
        pass
    
    def get_action(self, state, observation, reward):
        pass

    def update_policy(self, state, observation, action, reward, get_state = lambda x: 0):
        pass

class DeepQPolicy(Policy):
    """
    QPolicy for continuous state space (use neural net to estimate q function)
    """
    def __init__(self, env):
        from dql import DQLAgent
        self.gamma = .9
        #self.model = DQLAgent(env, self.gamma).model

    
    def get_action(self, state, observation, reward):
        pass

    def update_policy(self, state, observation, action, reward, get_state = lambda x: 0):
        pass

class QPolicy(Policy):
    """
    Q learning for discrete state space (find policy which maximizes total reward) 
    """
    def __init__(self, actions):
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))
        self.gamma = .9
        self.learning_rate = .1
        self.epsilon = .2 # probability of choosing a random movement in the policy
        
    def get_action(self, state, observation, reward):
        if random.uniform(0,1) < self.epsilon:
            return random.randint(0,3)
        else:
            rewards = self.q_table[state]
            return np.argmax(rewards) # return index of max reward, which corresponds to the action to take

    def update_policy(self, state, new_state, observation, action, reward):
        # get_state is a function that returns the current state, given an observation
        # q learning update rule
        self.q_table[state][action] = self.q_table[state, action] + self.learning_rate * (reward + self.gamma * np.max(self.q_table[new_state]) - self.q_table[state][action])

class Imitation(Policy):
    """
    give 20 examples of movement, then make it imitate
    """
    pass

class Dagger(Policy):
    """
    give examples of states, say what action to do
    """
    pass

class NaivePolicy(Policy):
    """
    If previous action had a reward above a threshold, then do that action again. Else, pick a random action
    """
    def __init__(self, actions):
        self.reward = 0
        self.action = 0
        self.epsilon = .2
        self.threshold = 0
    
    def get_action(self, state, observation, reward):
        if random.uniform(0,1) > self.epsilon and self.reward > self.threshold:
            return self.action
        else:
            return random.randint(0,3)

    def update_policy(self, state, new_state, observation, action, reward):
        self.reward = reward
        self.action = action


class BasicPolicy(Policy):
    """
    If landed, then stop moving. Else, apply left thrust
    """
    def __init__(self, actions):
        pass

    def get_action(self, state, observation, reward):
        if observation[6] == 1 and observation[7] == 1:
            return 0
        else:
            return 1

    def update_policy(self, state, observation, action, reward, get_state = lambda x: 0):
        self.reward = reward
        self.action = action

class RandomPolicy(Policy):
    """
    If landed, then stop moving. Else, apply left thrust
    """
    def __init__(self, actions):
        pass

    def get_action(self, state, observation, reward):
        return random.randint(0,3)

    def update_policy(self, state, observation, action, reward, get_state = lambda x: 0):
        pass