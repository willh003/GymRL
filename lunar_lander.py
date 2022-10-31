import gym
import random
import numpy as np

import policy

class Simulator:
    def __init__(self, task, policy):
        self.learning_rate = 5
        self.gamma = .9
        self.actions = ["nothing", "left", "center", "right"]
        self.policy = policy(self.actions) # specify which policy to use here
        self.action = random.randint(0, 3)
        self.reps = 600
        self.make_env(task)

    def make_env(self, task):
        env = gym.make(task, render_mode="human")
        observation, info = env.reset(seed=42)
        
        # Initial step is always random
        # This should probably be left up to the policy class
        observation, reward, terminated, truncated, info = env.step(self.action)    
        state = self.get_state(observation)

        for _ in range(self.reps):
            
            action = self.policy.get_action(state, observation, reward)
            new_observation, new_reward, terminated, truncated, info = env.step(action)
            new_state = self.get_state(new_observation)
            
            self.policy.update_policy(state, new_state, observation, action, reward)
            observation = new_observation
            state = new_state

        if terminated or truncated:
            observation, info = env.reset()
        env.close()

    def get_state(self, observation):
        return "not implemented"

if __name__ == "__main__":
    Simulator("LunarLander-v2", policy.NaivePolicy)
