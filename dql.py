import random
import numpy as np
from collections import deque
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizers import adam_v2

class DQLAgent:
    def __init__(self, env, gamma=0.99):
        self.env = env
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.01
        self.tot_reward = []
        self.batch_size = 1000
        self.memory = deque(maxlen=500000)
        self.osn = env.observation_space.shape[0]
        self.opt = adam_v2.Adam(learning_rate=0.001)
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        # Model maps from state space (continuous inputs, based on environment) to action space
        model.add(Dense(64, input_dim=self.osn, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.env.action_space.n, activation='linear'))
        model.compile(loss='mse', optimizer=self.opt)
        return model

    def act(self, state):
        if random.random() <= self.epsilon:
            return self.env.action_space.sample()
        # predict based on current environment inputs
        action = self.model.predict(state, verbose=0)
        return np.argmax(action[0])

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def replay_batch(self):
        # Choose a random sample from the memory to fit for this batch
        batch = random.sample(self.memory, self.batch_size)

        state = np.squeeze(np.array([i[0] for i in batch]))
        action = np.array([i[1] for i in batch])
        reward = np.array([i[2] for i in batch])
        next_state = np.squeeze(np.array([i[3] for i in batch]))
        done = np.array([i[4] for i in batch])

        q_val = reward + self.gamma * np.amax(self.model.predict_on_batch(next_state), \
                                            axis=1) * (1 - done)
        target = self.model.predict_on_batch(state)
        idx = np.arange(self.batch_size)
        target[[idx], [action]] = q_val

        # might not be enough info here. CHeck batch 1000 vs 64
        self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
