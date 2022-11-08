import random
import os
import numpy as np
from collections import deque
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.losses import MeanSquaredError
from tensorflow.python.keras import layers
from tensorflow.python.keras.optimizers import adam_v2
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

class ImitateAgent:
    def __init__(self):
        self.build_model()

    def load_data(self, filename):
        data_path = os.path.join("data", filename + ".csv")
        train = pd.read_csv(data_path)
        self.features = train.copy()
        self.labels = pd.concat([self.features.pop('vertical'), self.features.pop('lateral')], axis=1, join='inner')
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
    
    def build_model(self):
        # TODO: figure out how to normalize
        # normalize_layer = Normalization()
        # normalize_layer.adapt(self.features)
        model = Sequential([
                layers.Dense(16, activation='relu'),
                layers.Dense(2)
                ])

        model.compile(loss = MeanSquaredError(),
                      optimizer =adam_v2.Adam())
        
        self.model = model

    def train(self):
        print(self.features.shape)
        print("------------------")
        print(self.labels.shape)
        return self.model.fit(self.features, self.labels, epochs=25)

    def graph_loss(self, losses):
        plt.plot(range(1, len(losses) + 1), losses)
        plt.show()

if __name__=="__main__":
    agent = ImitateAgent()
    agent.load_data("trial01")
    agent.build_model()
    history = agent.train()
    losses = history.history['loss']
    arr= np.array( [0.054918479174375534,0.17031405866146088,0.07264575362205505,-0.15633781254291534,-0.29314637184143066,0.010642982088029385,0.0,0.0])
    pred = agent.model.predict(arr.reshape(1,8))
    print(pred, agent.model(arr.reshape(1,8)))
    agent.graph_loss(losses)

# should I just use two models instead (one for lateral, one for vertical)? Probably not, since they are probably not independent
# how do I get it to output 1x2 tensor, containing both vals