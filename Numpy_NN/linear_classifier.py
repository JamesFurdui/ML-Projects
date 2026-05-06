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
        verbose=False
    ):
        num_train, dim = X.shape
        num_classes = np.max(y) + 1

        if self.W is None:
            self.W = 0.001 * np.random.randh(dim, num_classes)

        loss_history = []

        for it in range(num_iters):
            # X_batch = None
            # y_batch = None

            batch_idx = np.random.choice(num_train, batch_size, replace=False)
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            
            loss, dW = softmax_loss_vectorized(W, X_batch, y_batch, reg)
            
            # loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            W -= learning_rate * dW

            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))
        
        return W, loss_history
    
    def loss(self, X_batch, y_batch, reg):

        pass