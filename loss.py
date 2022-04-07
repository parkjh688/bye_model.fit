import tensorflow as tf


class CustomAccuracy(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_pred - y_true))
        rmse = tf.math.sqrt(mse)
        return rmse / tf.reduce_mean(tf.square(y_true)) - 1