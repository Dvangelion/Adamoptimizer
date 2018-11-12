import tensorflow as tf
from adam import Adam
import matplotlib.pyplot as plt

if __name__ == '__main__':
    Adam = Adam()
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    #x_train = x_train[:100,:,:]
    #y_train = y_train[:100]
    print(y_train.shape)
    print(y_train[:10])
    print(x_train.shape)
    #print(Adam.one_hot_encoding(y_train))
    x_train, x_test = x_train / 255.0, x_test / 255.0

    train_costs,val_costs = Adam.model(x_train, y_train, x_test, y_test)

    plt.plot(range(len(train_costs)), train_costs)
    plt.show()

    plt.plot(range(len(val_costs)), val_costs)
    plt.show()



