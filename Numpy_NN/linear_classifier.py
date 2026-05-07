import os
from builtins import range
from builtins import object
import numpy as np
from softmax_function import *
from past.builtins import xrange

class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(
        self,
        X,
        y,
        learning_rate=1e-3,
        reg=1e-5,
        num_iters=100,
        batch_size=200,
        lr_decay=None,
        decay_interval=None,
        verbose=False
    ):
        num_train, dim = X.shape
        num_classes = np.max(y) + 1

        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)

        loss_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            X_batch = X[np.random.choice(num_train, batch_size, replace=False)]
            y_batch = y[np.random.choice(num_train, batch_size, replace=False)]
            
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            self.W -= learning_rate * grad

            # Verbose=True will print loss every 100 iterations
            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

            # Learning rate decay
            if lr_decay is not None and it % decay_interval == 0:
                learning_rate *= lr_decay
        
        return loss_history
    
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        scores = X.dot(self.W)
        y_pred = np.argmax(scores, axis=1)

        return y_pred
    
    def loss(self, X_batch, y_batch, reg):
        pass

    def save(self, fname):
        fpath = os.path.join("saved", fname)
        params = {"W": self.W}
        np.save(fpath, params)
        print(fname, "saved")

    def load(self, fname):
        fpath = os.path.join("saved", fname)
        if not os.path.exists(fpath):
            print(fname, "not available.")
            return False
        else:
            params = np.load(fpath, allow_pickle=True).item()
            self.W = params["W"]
            print(fname, "loaded.")
            return True
        
class Softmax(LinearClassifier):
    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_naive(self.W, X_batch, y_batch, reg)