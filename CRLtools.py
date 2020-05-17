"""
Continous space RL model
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import numpy as np
import random


MEMORY_SIZE = 1000000
BATCH_SIZE = 30


class IFRL(object):
    def __init__(self, nobs, nact):
        self.epsilon = 1
        self.nobs = nobs
        self.nact = nact
        self.gamma = 0.95
        self.activation = "sigmoid"
        self.lr = 1e-3
        self.epsilon_decay = 0.9
        self.TrainingData = deque(maxlen=MEMORY_SIZE)
        self.minExp = 0.008
        self.isimproved = False


    def setQfunc(self, Hnode):
        model = Sequential()
        model.add(Dense(Hnode[0], input_shape=(self.nobs,), activation=self.activation))
        for i in range(1, len(Hnode) - 1):
            model.add(Dense(Hnode[i], activation=self.activation))
        model.add(Dense(self.nact, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.lr))
        self.__Qfunc = model


    @property
    def Qfunc(self):
        return self.__Qfunc


    @Qfunc.setter
    def Qfunc(self, values):
        self.__Qfunc = values


    def PlanningD(self, state, action, reward, nstate, done):
        self.TrainingData.append((state, action, reward, nstate, done))


    def Policy(self, state):
        state = np.atleast_2d(state)
        if np.random.rand() < self.epsilon:
            return random.randrange(self.nact)
        return np.argmax(self.__Qfunc.predict(state)[0])


    def FitQ(self, alpha):
        if len(self.TrainingData) < BATCH_SIZE:
            return
        batch = random.sample(self.TrainingData, BATCH_SIZE)
        for state, action, reward, nstate, terminal in batch:
            nq = reward
            state, nstate = np.atleast_2d(state),np.atleast_2d(nstate)
            if not terminal:
                nq += self.gamma * np.max(self.__Qfunc.predict(nstate)[0])
            Q = self.__Qfunc.predict(state)
            Q[0][action] += alpha * (nq - Q[0][action])
            self.__Qfunc.fit(state, Q, verbose=0)
        self.epsilon = max(self.minExp, self.epsilon)




