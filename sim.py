import gym
import random
import numpy as np
import policy
import matplotlib.pyplot as plt
import sys
import os

class Simulator:
    def __init__(self, task, policy):
        self.learning_rate = 5
        self.gamma = .9
        self.actions = ["nothing", "left", "center", "right"]
        self.policy = policy(self.actions) # specify which policy to use here
        self.action = random.randint(0, 3)
        self.reps = 600
        self.task = task

    def make_env(self):
        env = gym.make(self.task, render_mode="human")
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

    def train_q(self, episodes):
        agent = self.learn_env(episodes)
        self.rewards = agent.tot_reward
        self.model = agent.model

    def learn_env(self, episodes):
        from dql import DQLAgent # Slow import, not necessary unless using q learning
        env = gym.make(self.task, render_mode="human")
        max_steps = 350
        agent = DQLAgent(env, self.gamma, max_steps)
        for e in range(1, episodes + 1):
            state, info = env.reset() # seed it here for testing
            state = np.reshape(state, [1, agent.osn]) # reshape env to inputs of DQL network
            t_reward = 0
            
            for step in range(max_steps):
                # TODO: find a way to penalize taking more steps
                action = agent.act(state)
                next_state, reward, done, truncated, info = env.step(action)
                next_state = np.reshape(next_state, [1, agent.osn])
                agent.memorize(state, action, reward, next_state, done, step)
                state = next_state
                t_reward += reward
                #print(len(agent.memory))
                if len(agent.memory) > agent.batch_size:
                    agent.replay_batch()
                if done:

                    break
                                
            print(f'Episode: {e} | Steps: {step} | Total reward: {t_reward} | Epsilon: {agent.epsilon}')
            agent.tot_reward.append(t_reward)
        
        return agent

    def save_model(self, model_name):
        rewards = self.rewards
        plt.plot(range(1, len(rewards) + 1), rewards)
        fig_path = os.path.join("models", "rewards", model_name)
        plt.savefig(fig_path)
        model_path = os.path.join("models", model_name)
        self.model.save(model_path)


    def get_state(self, observation):
        return "not implemented"



if __name__ == "__main__":
    episodes = int(sys.argv[1])
    save_name = sys.argv[2]
    if len(sys.argv) > 3:
        task = sys.argv[3]
    else:
        task = "LunarLander-v2"
    sim = Simulator(task, policy.QPolicy)
    sim.train_q(episodes)
    sim.save_model(save_name)
