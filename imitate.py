import random
import os
import numpy as np
from collections import deque
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.losses import MeanSquaredError
from tensorflow.python.keras import layers
from tensorflow.python.keras.optimizers import adam_v2
import pandas as pd

class ImitateAgent:
    def __init__(self, gamma=0.99, max_steps=1000):
        self.build_model()

    def load_data(self, filename):
        data_path = os.path.join("data", filename + ".csv")
        train = pd.read_csv(data_path)
        self.features = train.copy()
        self.labels = pd.concat([self.features.pop('vertical'), self.features.pop('lateral')])
    
    def build_model(self):
        # TODO: figure out how to normalize
        # normalize_layer = Normalization()
        # normalize_layer.adapt(self.features)
        model = Sequential([
                layers.Dense(64),
                layers.Dense(1)
                ])

        model.compile(loss = MeanSquaredError(),
                      optimizer =adam_v2.Adam())
        
        self.model = model

    def train(self):
        self.model.fit(self.features, self.labels, epochs=10)

if __name__=="__main__":
    agent = ImitateAgent()
    agent.load_data("trial01")
    agent.build_model()
    agent.train()
    print(agent.features.head())
