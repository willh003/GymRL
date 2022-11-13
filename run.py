from tensorflow.python.keras.models import load_model
import gym
import numpy as np
import sys
import os

def run_model(task, model_name):
    model = load_model(model_name)
    env = gym.make(task, render_mode="human")
    state, info = env.reset()
    done = False

    while not done:
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        print(state)
        print(env.observation_space.shape)
        action = np.argmax(model.predict(state, verbose=0)[0])
        env.render()
        state, reward, done, truncated, info = env.step(action)
    env.close()

if __name__ == "__main__":
    task = "LunarLander-v2"
    model_path = os.path.join("models", sys.argv[1], sys.argv[2])
    run_model(task, model_path)