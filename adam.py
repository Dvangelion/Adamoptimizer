import numpy as np


class Adam:
    def __init__(self):
        #step size
        self.alpha = .001

        #exponential decays
        self.beta_1 = .9
        self.beta_2 = .999

        self.num_iterations = 200
        self.batch_size = 128

    def sigmoid(self, x):
        y = 1.0 / (1.0 + np.exp(-x))
        return y

    def softmax(self, z):
        z_exp = np.exp(z)
        softmax = z_exp/(sum(z_exp))
        return softmax

    def one_hot_encoding(self, y):
        #m, n = y.shape
        length = len(y)
        categories = 10
        one_hot = np.zeros((length, categories))
        for i in range(length):
            one_hot[i][y[i]] = 1

        return one_hot



    def compute_grad(self, W, b, X, Y):

        m = X.shape[0]

        logits = np.dot(W.T,X) + b
        O = self.softmax(logits)
        Y = self.one_hot_encoding(Y).T

        cost = -(1.0/m)*np.sum(Y*np.log(O) + (1.0-Y)*np.log(1.0-O))

        dw = -(1.0/m)*np.dot(X, (O-Y).T)
        db = -(1.0/m)*np.sum(O-Y)

        grads = {"dw":dw, "db":db}

        return grads, cost

    def adam_optimize(self,W, b, X, Y):

        X = X.T
        alpha = self.alpha
        beta_1 = self.beta_1
        beta_2 = self.beta_2
        num_iterations = self.num_iterations
        momentum_w = 0
        momentum_b = 0
        v_w = 0
        v_b = 0
        eps = 1e-8
        costs = []
        iter = 0

        while  iter < num_iterations:
            iter += 1
            grads, cost = self.compute_grad(W,b,X,Y)
            grads_w = grads['dw']
            grads_b = grads['db']
            costs.append(cost)

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
            W += dW
            b += db

        params = {"W":W, "b":b}
        return params, costs

    def optimize(self, W, b, X, Y):
        X = X.T
        costs = []
        iter = 0
        while iter < self.num_iterations:
            iter += 1
            grads, cost = self.compute_grad(W, b, X, Y)
            costs.append(cost)
            dw = grads['dw']
            db = grads['db']

            W += self.alpha*dw
            b += self.alpha*db


        params = {"W":W, "b":b}

        return params, costs



    def predict(self, W, b, X):
        X = X.T

        logit = self.softmax(np.dot(W.T, X) + b)
        y_pred = np.argmax(logit, axis=0)


        return y_pred



    def model(self, X_train, Y_train, X_test, Y_test):
        #TODO: add mini batches
        #initialize parameterss
        W = np.random.rand(784,10)
        b = np.random.rand(10,1)

        X_train = X_train.reshape(-1,784)
        X_test = X_test.reshape(-1,784)

        X = X_train
        Y = Y_train


        index = len(X_train)//5

        train_costs_list = []
        val_costs_list = []

        for i in range(5):
            p = np.random.permutation(len(X))
            X = X[p]
            Y = Y[p]

            X_val = X[:index,:]
            Y_val = Y[:index]
            X_train = X[index:,:]
            Y_train = Y[index:]


            params, train_costs = self.optimize(W, b, X_train, Y_train)
            val_params, val_costs = self.optimize(W, b, X_val, Y_val)

            print(train_costs)
            print(val_costs)


            train_costs_list.append(train_costs)
            val_costs_list.append(val_costs)



        train_costs = np.array(train_costs_list)
        val_costs = np.array(val_costs_list)

        train_costs = sum(train_costs)/10
        print(train_costs)
        val_costs = sum(val_costs)/10
        print(val_costs)
        W = params['W']
        b = params['b']

        y_pred_train = self.predict(W,b,X_train)
        y_pred_test = self.predict(W,b,X_test)


        train_acc = sum(y_pred_train == Y_train) / len(Y_train)
        test_acc = sum(y_pred_test == Y_test) / len(Y_test)
        val_acc = sum()

        print ("Train acc:", train_acc)
        print ("Test acc:", test_acc)

        return train_costs, val_costs


if __name__ == '__main__':
    Adam = Adam()
    print(Adam.softmax(np.zeros((3,4))))


