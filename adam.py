import numpy as np


class Adam:
    def __init__(self):
        #step size
        self.alpha = .001

        #exponential decays
        self.beta_1 = .9
        self.beta_2 = .999

        self.num_iterations = 10
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

    def optimize(self, W, b, X, Y, batch_size):
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
        total_batches = len(Y) // batch_size
        while iter < num_iterations:
            iter += 1
            i = 0
            cost = 0
            p = np.random.permutation(len(X.T))
            X = X[:, p]
            Y = Y[p]

            while i < total_batches:

                batch_X = X[:, i*batch_size:(i+1)*batch_size]
                batch_Y = Y[i*batch_size:(i+1)*batch_size]

                i += 1
                grads, batch_cost = self.compute_grad(W, b, batch_X, batch_Y)
                grads_w = grads['dw']
                grads_b = grads['db']
                costs.append(batch_cost)

                alpha = alpha / np.sqrt(iter)

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

            #costs.append(cost/total_batches)


        costs = np.array(costs)
        params = {"W":W, "b":b}

        return params, costs

    def grad_optimize(self, W, b, X, Y):
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
        
        #5-fold cross validation
        index = len(X_train) // 5
        
        X_val = X[:index,:]
        Y_val = Y[:index]
        X_train = X[index:,:]
        Y_train = Y[index:]

        train_costs_list = []
        train_acc_list = []
        val_costs_list = []
        val_acc_list = []

        for i in range(5):





            params, train_costs = self.optimize(W, b, X_train, Y_train, self.batch_size)
            val_params, val_costs = self.optimize(W, b, X_val, Y_val, self.batch_size)

            train_costs_list.append(train_costs)
            val_costs_list.append(val_costs)

            y_pred_train = self.predict(W, b, X_train)
            y_pred_val = self.predict(W,b, X_val)
            train_acc = sum(y_pred_train == Y_train) / len(Y_train)
            val_acc = sum(y_pred_val == Y_val) / len(Y_val)

            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)



        train_costs = np.array(train_costs_list)
        val_costs = np.array(val_costs_list)

        print ("train_costs", train_costs.shape)
        train_costs = np.sum(train_costs,axis=0)/5
        print(train_costs)
        print(train_costs.shape)
        val_costs = np.sum(val_costs,axis=0)/5
        #print(val_costs)
        W = params['W']
        b = params['b']



        train_acc = sum(train_acc_list) / len(train_acc_list)
        val_acc = sum(val_acc_list) / len(val_acc_list)

        y_pred_test = self.predict(W,b,X_test)
        test_acc = sum(y_pred_test == Y_test) / len(Y_test)


        print ("Train acc:", train_acc)
        print ("Val acc:", val_acc)
        print ("Test acc:", test_acc)

        return train_costs, val_costs


if __name__ == '__main__':
    Adam = Adam()
    print(Adam.softmax(np.zeros((3,4))))


