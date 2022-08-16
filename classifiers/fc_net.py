from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        self.D = input_dim
        self.H = hidden_dim
        self.C = num_classes

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        W1 = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        W2 = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        b1 = np.zeros(hidden_dim)
        b2 = np.zeros(num_classes)

        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        W1, W2, b1, b2 = self.params['W1'], self.params['W2'], self.params['b1'], self.params['b2']

        # X = X.reshape(X.shape[0], self.D)

        fc1, (X1, W1, b1) = affine_forward(X, W1, b1) #cache = (x,w,b)

        # fc1 = X.dot(W1) + b1
        X2, fc1 = relu_forward(fc1)
        # X2 = np.maximum(0, fc1)
        # scores = X2.dot(W2) + b2
        scores, (X2, W2, b2) = affine_forward(X2, W2, b2)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        reg = self.reg
        N = X.shape[0]

        # softmax loss #########################################################
        # assure numeric stability
        # f = scores - np.max(scores, axis=1).reshape(-1,1)

        # softmax = np.exp(f) / np.sum( np.exp(f), axis=1).reshape(-1,1)
        # loss = -np.log(softmax[np.arange(N), y])
        # loss = np.sum(loss)
        # loss /= N

        ## using the function already in place:
        loss, softmax = softmax_loss(scores, y)
        # 0.5 as in the instructions
        loss += 0.5 * reg * (np.sum(W2 * W2) + np.sum( W1 * W1 )) # regularization

        # gradients ############################################################
        # ## dL/dy_hat (see https://stackoverflow.com/a/73065678/7060954)
        # ### -1 + e^(fi)/sum(e^(fi))  if j=yi
        # ###      e^(fi)/sum(e^(fi))  otherwise
        # softmax[np.arange(N), y] -= 1
        # softmax /= N

        # Second layer derivatives #############################################
        # # dL/dW2 = dL/dy_hat * dy_hat/dW2
        # # shapes: W2 = [H*C]; score=softmax= [N*C]; X=[N*H]
        # dW2 = X2.T.dot(softmax) # [H*N][N*C] = [H*C] => W2 or dW2 dims
        # db2 = np.sum(softmax, axis=0) # [C]
        dx2, dW2, db2 = affine_backward(softmax, (X2, W2, b2))

        # Relu derivative ######################################################
        # dL/dW1 => dL/dy_hat * dy_hat/dX2 * dX2/dW1 
        # dx2 = softmax.dot(W2.T) # [N*C][C*H] = [N*H]
        # max(0,fc1) -> derived: 1(fc1 >= 0)
        # dfc1 = fc1 > 0 # dX2/dfc1
        # dfc1 = dx2 * dfc1 # upstream * dfc1 # [N*H]*[N*H] = [N*H]
        dfc1 = relu_backward(dx2, (fc1))

        # First layer derivatives ##############################################
        ## target shape like W = [D*H]
        # dW1 = X.T.dot(dfc1) # [D*N]*[N*H] = [D*H]
        # db1 = np.sum(dfc1, axis=0) # [H]
        dx1, dW1, db1 = affine_backward(dfc1, (X1, W1, b1))

        # regularization
        dW1 += reg * W1
        dW2 += reg * W2

        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

        # dscores = softmax
        # dscores[range(N),y] -= 1
        # dscores /= N

        # # W2 and b2
        # grads['W2'] = np.dot(X2.T, dscores)
        # grads['b2'] = np.sum(dscores, axis=0)
        # # next backprop into hidden layer
        # dhidden = np.dot(dscores, W2.T)
        # # backprop the ReLU non-linearity
        # dhidden[X2 <= 0] = 0
        # # finally into W,b
        # grads['W1'] = np.dot(X.T, dhidden)
        # grads['b1'] = np.sum(dhidden, axis=0)

        # # add regularization gradient contribution
        # grads['W2'] += reg * W2
        # grads['W1'] += reg * W1

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
