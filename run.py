from tensorflow.python.keras.models import load_model
import gym
import numpy as np
import sys

def run_model(task, model_name):
    model = load_model(model_name)
    env = gym.make(task, render_mode="human")
    state, info = env.reset()
    done = False

    while not done:
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        action = np.argmax(model.predict(state, verbose=0)[0])
        env.render()
        state, reward, done, truncated, info = env.step(action)
    env.close()

if __name__ == "__main__":
    task = "LunarLander-v2"
    model_path = sys.argv[1]
    run_model(task, model_path)