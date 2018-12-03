import tensorflow as tf
from optimize import Optimize
import matplotlib.pyplot as plt

if __name__ == '__main__':
    Optimize = Optimize()
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    #x_train = x_train[:100,:,:]
    #y_train = y_train[:100]
    #print(y_train.shape)
    #print(y_train[:10])
    #print(x_train.shape)
    #print(Adam.one_hot_encoding(y_train))
    x_train, x_test = x_train / 255.0, x_test / 255.0

    train_costs,val_costs, test_costs = Optimize.model(x_train, y_train, x_test, y_test, method = 'adam')
    r_tr, r_v, r_t = Optimize.model(x_train, y_train, x_test, y_test, method = 'rmsprop')
    g_tr, g_v, g_t = Optimize.model(x_train, y_train, x_test, y_test, method = 'grad')

    plt.plot(range(len(train_costs)), train_costs, label='Adam')
    plt.plot(range(len(train_costs)), r_tr, label='RMSProp')
    plt.plot(range(len(train_costs)), g_tr, label = 'Gradient')
    plt.legend()
    plt.grid()
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.title('Batch training cost over iterations')
    plt.show()

    plt.plot(range(len(val_costs)), val_costs, label='Adam')
    plt.plot(range(len(val_costs)), r_v, label='RMSProp')
    plt.plot(range(len(val_costs)), g_v, label = 'Gradient')
    plt.legend()
    plt.grid()
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.title('Batch validation cost over iterations')
    plt.show()

    plt.plot(range(len(test_costs)), test_costs, label='Adam')
    plt.plot(range(len(test_costs)), r_t, label='RMSProp')
    plt.plot(range(len(test_costs)), g_t, label = 'Gradient')
    plt.legend()
    plt.grid()
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.title('Batch test cost over iterations')
    plt.show()



