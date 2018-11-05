#import tensorflow as tf
import numpy as np

#mnist = tf.keras.datasets.mnist

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train, x_test = x_train / 255.0, x_test / 255.0


class Adam:
    def __init__(self):
        #step size
        self.alpha = .001

        #exponential decays
        self.beta_1 = .9
        self.beta_2 = .999

        self.num_iterations = 1000

    def sigmoid(self, x):
        y = 1.0 / (1.0 + np.exp(-z))
        return y

    def propagate(self,W,b,X,Y):
        m = X.shape[1]
        z = np.dot(W.T,X) + b
        A = sigmoid(z)
        cost = -1.0/m*np.sum(Y*np.log(A) + (1.0-Y)*np.log(1.0-A))

        dw = 1.0/m*np.dot(X, (A-Y).T)
        db = 1.0/m*np.sum(A-Y)

        cost = np.squeeze(cost)
        grads = {"dw":dw, "db":db}

        return grads, cost

    def compute_grad(self,W,b,X,Y):

        m = X.shape[1]
        z = np.dot(W.T,X) + b
        A = sigmoid(z)

        dw = 1.0/m*np.dot(X,(A-Y).T)
        db = 1.0/m*np.sum(A-Y)

        grads = {"dw":dw, "db":db}

        return grads

    def optimize(self,W, b, X, Y):
        alpha = self.alpha
        beta_1 = self.beta_1
        beta_2 = self.beta_2
        momentum_w = 0
        momentum_b = 0
        v_w = 0
        v_b = 0
        eps = 1e-8
        num_iterations = self.num_iterations

        iter = 0
        while  iter < num_iterations:
            iter += 1
            grads_w, grads_b = self.compute_grad(W,b,X,Y)['dw'], compute_grad(W,b,X,Y)['db']

            #Update biased first moment estimate
            momentum_w = beta_1*momentum_w + (1-beta_1)*grads_w
            momentum_b = beta_1*momentum_b + (1-beta_1)*grads_b

            #Update biased second raw moment estimate
            v_w = beta_2*v_w + (1-beta_2)*(grads_w*grads_w)
            v_b = beta_2*v_b + (1-beta_2)*(grads_b*grads_b)

            #Compute bias-corrected first moment estimate
            m_hat_w = momentum_w/(1-beta_1**iter)
            m_hat_b = momentum_b/(1-beta_1**iter)

            #Compute bias-corrected second raw moment estimate
            v_hat_w = v_w/(1-beta_2**iter)
            v_hat_b = v_b/(1-beta_2**iter)

            #Compute parameters to update
            dW = alpha*m_hat_w/(np.sqrt(v_hat_w)+eps)
            db = alpha*m_hat_b/(np.sqrt(v_hat_b)+eps)

            #Update parameters
            W -= dW
            b -= db

        params = {"W":W, "b":b}
        return params

    def predict(self, W,b,X):

        m = X.shape[1]
        y_pred = np.zeros((1,m))
        W = W.reshape(X.shape[0],1)

        A = self.sigmoid(np.dot(W.T, X) + b)
        for i in range(A.shape[1]):
            if A[:,i] > .5:
                y_pred[:,i] = 1
            elif A[:,i] <= .5:
                y_pred[:,i] = 0

        return y_pred

    def model(self,X_train, Y_train, X_test, Y_test):

        #initialize parameterss
        W = np.zeros((X_train.shape[0], 1))
        b = 0

        params = self.optimize(W, b, X_train, Y_train)
        W = params['W']
        b = params['b']

        y_pred_train = self.predict(W,b,X_train)
        y_pred_test = self.redict(W,b,X_test)

        train_acc = sum(y_pred_train == Y_train) / len(Y_train)
        test_acc = sum(y_pred_test == Y_test) / len(Y_test)

        print ("Train acc:", train_acc)
        print ("Test acc:", test_acc)

        return test_acc



