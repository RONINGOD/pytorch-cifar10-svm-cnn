import numpy as np
from svm import SupportVectorMachine
from tqdm import tqdm

class SupportVectorClassifier(object):
    def __init__(self, iteration=100, penalty=1.0, epsilon=1e-6, kernel=None):
        self.iteration = iteration
        self.penalty = penalty
        self.epsilon = epsilon
        self.kernel = kernel
        self.classifier = []

    def __build_model(self, y):
        self.label = np.unique(y)
        for i in range(len(self.label)):
            for j in range(i+1, len(self.label)):
                model = SupportVectorMachine(self.iteration, self.penalty, self.epsilon, self.kernel)
                self.classifier.append((i, j, model))

    def fit(self, X, y):
        self.__build_model(y)
        for i, j, model in tqdm(self.classifier,desc='Training {}'.format(self.kernel['name'])):
            index = np.where((y == self.label[i]) | (y == self.label[j]))[0]
            X_ij, y_ij = X[index], np.where(y[index] == self.label[i], -1, 1)
            model.fit(X_ij, y_ij)
    
    def predict(self, X):
        vote = np.zeros((X.shape[0], len(self.label)))
        for i, j, model in tqdm(self.classifier,desc='Predicting {}'.format(self.kernel['name'])):
            y = model.predict(X)
            vote[np.where(y == -1)[0], i] += 1
            vote[np.where(y == 1)[0], j] += 1
        return self.label[np.argmax(vote, axis=1)]
