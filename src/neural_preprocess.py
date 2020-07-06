import numpy as np


class Input:
    
    def __init__(self, X, num_atoms=23):
        self.step = 1.0
        self.noise = 1.0
        self.triuind = (np.arange(num_atoms)[:, np.newaxis] <= np.arange(num_atoms)[np.newaxis, :]).flatten()
        self.max = 0
        for _ in range(10): self.max = np.maximum(self.max, self.realize(X).max(axis=0))
        X = self.expand(self.realize(X))
        self.mean = X.mean(axis=0)
        self.std = (X - self.mean).std()

    def realize(self, X):
        def _realize_(x):
            inds = np.argsort(-(x ** 2).sum(axis=0) ** .5 + np.random.normal(0, self.noise, x[0].shape))
            x = x[inds, :][:, inds] * 1
            x = x.flatten()[self.triuind]
            return x
        return np.array([_realize_(z) for z in X])

    def expand(self, X):
        Xexp = []
        for i in range(X.shape[1]):
            for k in np.arange(0, self.max[i] + self.step, self.step):
                Xexp += [np.tanh((X[:, i] - k) / self.step)]
        return np.array(Xexp).T

    def normalize(self, X):
        return (X - self.mean) / self.std

    def forward(self, X):
        return self.normalize(self.expand(self.realize(X))).astype('float32')
