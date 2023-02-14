import numpy as np
import random
import json
from copy import deepcopy


class GRU_RNN:

    def __init__(self, T, learning_rate=0.0001, k1=10, k2=12, grad_threshold=0.0001, weight_file=None):
        """
        T : length of input sequence
        learning_rate : weights the influence of one gradient update
        k1 : length of forward pass in TBPTT
        k2 : length of backward pass in TBPTT
        weight_file : json-file from which pre-computed weights are loaded for initialization - None results in random initialization
        """

        self.learning_rate = learning_rate
        self.T = T
        self.k1 = k1    # lenght of forward pass
        self.k2 = k2    # lenght of backpropagation
        self.gradient_threshold = grad_threshold

        if weight_file:
            with open('weights.json', 'r') as fp:
                self.weights = json.load(fp)
        else:
            np.random.seed(662)                 # make results reproducible
            self.weights = {
                'W_rh': np.random.uniform(),    # input and output dimension are both 1, therefore all weights are float
                'W_rx': np.random.uniform(),
                'b_r': np.random.uniform(),
                'W_zh': np.random.uniform(),
                'W_zx': np.random.uniform(),
                'b_z': np.random.uniform(),
                'W_hh': np.random.uniform(),
                'W_hx': np.random.uniform(),
                'b_h': np.random.uniform(),
                'h_0': np.random.uniform()      # note that for the first point of the sequence, we need an artificial hidden state, which is another parameter to be trained
            }

        self.best_weights = deepcopy(self.weights)

    
    def GRU_cell(self, h_old, x, weights):
        # Reset gate
        r_pre = weights['W_rh'] * h_old + weights['W_rx'] * x + weights['b_r']
        r = sigma(r_pre)

        # Update gate
        z_pre = weights['W_zh'] * h_old + weights['W_zx'] * x + weights['b_z']
        z = sigma(z_pre)

        # Proposing updated hidden state
        h_tilde_pre = weights['W_hh'] * r * h_old + weights['W_hx'] * x + weights['b_h']
        h_tilde = tanh(h_tilde_pre)

        # Compute updated hidden state
        h = (1.0 - z) * h_old + z * h_tilde

        # Output
        y = sigma(h)

        return (h_old, r_pre, r, z_pre, z, h_tilde_pre, h_tilde, h, y)
    

    def TBPTT(self, x, y):
        """Truncated Backpropagation Through Time. Updates the gradients for one whole sequence.
        x : input sequence of length T
        y : ground truth sequence of length T"""

        results = []
        h = self.weights['h_0']
        
        for t in range(self.T):
            ### Feedforward step
            results.append(self.GRU_cell(h, x[t], self.weights))    # (h_old, r_pre, r, z_pre, z, h_tilde_pre, h_tilde, h, y)
            h = results[-1][7]
            
            if (t+1) % self.k1 == 0:
                ### Backward pass through k2 time steps
                if t+1 == self.k1:
                    stop = -1   # k2 > k1, so in the first backpropagation we only do k1 time steps
                else:
                    stop = t-self.k2

                # Set gradients to zero
                nabla_W_rh = 0.0
                nabla_W_rx = 0.0
                nabla_b_r = 0.0
                nabla_W_zh = 0.0
                nabla_W_zx = 0.0
                nabla_b_z = 0.0
                nabla_W_hh = 0.0
                nabla_W_hx = 0.0
                nabla_b_h = 0.0

                dL_dhold = 0.0

                for tt in range(t,stop,-1):
                    # Get data for gradients
                    h_old, r_pre, r, z_pre, z, h_tilde_pre, h_tilde, h_new, y_pred = results[tt]

                    # Calculate gradients
                    dL_dh = dL_dhold + (y_pred - y[tt]) * sigma_prime(h_new)    # L(y_pred_t, y_t) = 0.5 * (y_pred - y)^2

                    dh_dz = - h_old + h_tilde
                    dz_db = sigma_prime(z_pre)
                    nabla_b_z_t = dL_dh * dh_dz * dz_db
                    nabla_b_z += nabla_b_z_t
                    nabla_W_zh += nabla_b_z_t * h_old
                    nabla_W_zx += nabla_b_z_t * x[tt]

                    dh_dhtilde = z
                    dhtilde_db = tanh_prime(h_tilde_pre)
                    nabla_b_h_t = dL_dh * dh_dhtilde * dhtilde_db
                    nabla_b_h += nabla_b_h_t
                    nabla_W_hh += nabla_b_h_t * r * h_old
                    nabla_W_hx += nabla_b_h_t * x[tt]

                    dhtilde_dr = tanh_prime(h_tilde_pre) * self.weights['W_hh'] * h_old
                    dr_db = sigma_prime(r_pre)
                    nabla_b_r_t = dL_dh * dh_dhtilde * dhtilde_dr * dr_db
                    nabla_b_r += nabla_b_r_t
                    nabla_W_rh += nabla_b_r_t * h_old
                    nabla_W_rx += nabla_b_r_t * x[tt]

                    dz_dhold = sigma_prime(z_pre) * self.weights['W_zh']
                    dh_dhold = 1 - z - h_old*dz_dhold + z*(tanh_prime(h_tilde_pre)*self.weights['W_hh']*(r + h_old*sigma_prime(r_pre)*self.weights['W_rh'])) + h_tilde*dz_dhold
                    dL_dhold = dL_dh * dh_dhold

                    if tt == 0:
                        nabla_h_0 = dL_dhold

                ### Update weights
                if stop > -1:
                    eta = self.learning_rate / self.k2  # we did k2 backpropagation iterations
                    thresh = self.gradient_threshold / self.k2
                else:
                    eta = self.learning_rate / self.k1  # we did k1 backpropagation iterations
                    thresh = self.gradient_threshold / self.k1

                # Clipping gradient
                G = np.array([nabla_W_rh, nabla_W_rx, nabla_b_r, nabla_W_zh, nabla_W_zx, nabla_b_z, nabla_W_hh, nabla_W_hx, nabla_b_h, nabla_h_0])
                norm_G = np.linalg.norm(G)
                if norm_G > thresh:
                    [nabla_W_rh, nabla_W_rx, nabla_b_r, nabla_W_zh, nabla_W_zx, nabla_b_z, nabla_W_hh, nabla_W_hx, nabla_b_h, nabla_h_0] = thresh/norm_G * G
                else:
                    [nabla_W_rh, nabla_W_rx, nabla_b_r, nabla_W_zh, nabla_W_zx, nabla_b_z, nabla_W_hh, nabla_W_hx, nabla_b_h, nabla_h_0] = eta * G
                
                self.weights['W_rh'] -= nabla_W_rh
                self.weights['W_rx'] -= nabla_W_rx
                self.weights['b_r'] -= nabla_b_r
                self.weights['W_zh'] -= nabla_W_zh
                self.weights['W_zx'] -= nabla_W_zx
                self.weights['b_z'] -= nabla_b_z
                self.weights['W_hh'] -= nabla_W_hh
                self.weights['W_hx'] -= nabla_W_hx
                self.weights['b_h'] -= nabla_b_h
                self.weights['h_0'] -= nabla_h_0

        return

    
    def feedforward(self, x):
        """Runs the RNN for the sequence x and returns the output y."""
        y = np.zeros(self.T)
        h = self.weights['h_0']
        for t in range(self.T):
            _, _, _, _, _, _, _, h, yt = self.GRU_cell(h, x[t], self.weights)
            y[t] = yt
        return y
    

    def evaluate(self, testing_data):
        """Evaluates the average loss per prediction.
        testing_data : list of (x,y)-tuples"""
        L = 0.0     # loss
        for (x, y) in testing_data:
            y_pred = self.feedforward(x)
            diff = y_pred - y
            L += 0.5 * np.dot(diff, diff)   # L(y_pred_t, y_t) = 0.5 * (y_pred - y)^2

        return L / len(testing_data)
    

    def train_network(self, training_data, epochs, testing_data=None):
        av_loss_best = float('inf')
        av_loss_list = []

        for epoch in range(epochs):
            # Shuffle training data
            random.shuffle(training_data)

            for x, y in training_data:
                self.TBPTT(x, y)    # truncated backpropagation through time for each xy-pair in the training data set

            # Evaluate permormance of updated weights on an independent test data set
            if testing_data:
                av_loss = self.evaluate(testing_data)
                av_loss_list.append(av_loss)
                print('Epoch ' + str(epoch+1) + '. Average loss on testing data: ' + str(av_loss))

                update_best_weights = False
                if av_loss < av_loss_best:
                    update_best_weights = True
                    av_loss_best = av_loss
            else:
                print('Epoch', epoch+1, 'complete.')
                update_best_weights = True

            # Update best weights
            if update_best_weights:
                self.best_weights.update(self.weights)
        
        # Store weights to file
        with open('weights.json', 'w') as fp:
            json.dump(self.best_weights, fp)
        
        return av_loss_list


    def predict(self, x):
        """Predicts the next T points for a sequence of T points using the best weights."""
        T = len(x)      # the input sequence does not need to have length self.T
        y = np.zeros(T)
        h = self.best_weights['h_0']
        for t in range(T):
            _, _, _, _, _, _, _, h, yt = self.GRU_cell(h, x[t], self.best_weights)
            y[t] = yt
        return y



# Auxiliary functions
def sigma(x):
    """The ReLU function."""
    if x <= 0.0:
        return 0.0
    else:
        return x

def sigma_prime(x):
    """Derivative of the ReLU function."""
    if x <= 0.0:
        return 0.0
    else:
        return 1.0

def tanh(x):
    """The hyperbolic tangent."""
    return 1.0 - 2.0/(np.exp(2*x) + 1)

def tanh_prime(x):
    """Derivative of the hyperbolic tangent."""
    return 1.0 - tanh(x)**2