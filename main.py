import tensorflow as tf
from adam import Adam

if __name__ == '__main__':
    Adam = Adam()
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    d = Adam.model(x_train, y_train, x_test, y_test)

