from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W) # -> 500x10
    loss = 0.0
    dW = np.zeros(W.shape)

    rows, cols = scores.shape
    for i in range(rows):
      yi = y[i]
      f = scores[i] - np.max(scores[i])
      softmax = np.exp(f) / np.sum(np.exp(f))
      loss += -np.log( softmax[yi] )

      # dW = dL/df * df/dW
      # where f = softmax
      ## if f(j) == f(yi): dL/df = -1 * f
      ## else f(j) : dL/df = f
      ### source: http://intelligence.korea.ac.kr/jupyter/2020/06/30/softmax-classifer-cs231n.html
      ### and my post: https://stackoverflow.com/a/73065678/7060954

      # further:
      ## dL/df * df/dW =
      ### -1*f * X [case j = yi] or 
      ### f*X [case j != yi]

      ### Post: https://stackoverflow.com/a/73065678/7060954
      for j in range(cols):
        if j != yi:
          dW[:,j] += softmax[j]*X[i]
        else:
          dW[:,j] += (softmax[j]-1)*X[i]

    # average the loss and scale the gradient
    loss /= rows
    dW /= rows

    # impose regularization
    loss += reg * np.sum(W * W) 
    dW += reg * 2 * W 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    N = X.shape[0]

    scores = X.dot(W) # shape = 500x10
    f = scores - np.max(scores, axis=1).reshape(-1,1)#, keepdims=True)
    softmax = np.exp(f) / np.sum( np.exp(f), axis=1).reshape(-1,1)#, keepdims=True)
    loss = -np.log(softmax[np.arange(N), y])
    loss = np.sum(loss)
    # sum_exp_scores = np.exp(scores).sum(axis=1, keepdims=True)
    # softmax = np.exp(scores)/sum_exp_scores
    # loss = np.sum(-np.log(softmax[np.arange(num_train), y]) )

    softmax[np.arange(N), y] -= 1
    dW += X.T.dot(softmax)

    # average out
    loss /= N
    dW /= N

    # impose regularization
    loss += reg * np.sum(W * W)
    dW += 2*W*reg

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
