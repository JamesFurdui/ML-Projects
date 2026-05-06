import numpy as np

def softmax_loss_naive(W, X, y, reg):
    """
    W: Numpy array containing weights
    X: Numpy array containing batch of data
    y: Numpy array containing ground truth labes [0, 1]
    reg: float containing regularization strength
    """

    loss = 0.0
    dW = np.zeros_like(W)

    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)

        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()
        logp = np.log(p)

        loss -= logp[y[i]]

        p[y[i]] -= 1
        dW += np.outer(X[i], p)

    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + 2 * reg * W
    
    return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
    """
    W: Numpy array containing weights
    X: Numpy array containing batch of data
    y: Numpy array containing ground truth labes [0, 1]
    reg: float containing regularization strength
    """

    loss = 0.0
    dW = np.zeros_like(W)

    num_train = X.shape[0]

    scores = X.dot(W)
    scores -= scores.max(axis=1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=1, keepdims=True)
    logp = np.log(p)

    loss -= np.sum(logp[np.arange(num_train), y])

    p[np.arange(num_train), y] -= 1
    dW += X.T.dot(p)
    
    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + 2 * reg * W

    return loss, dW
