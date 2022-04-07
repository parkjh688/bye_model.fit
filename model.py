import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import VGG16


class YogaPose(tf.keras.Model):
    def __init__(self, num_classes=30, freeze=False):
        super(YogaPose, self).__init__()
        self.base_model = EfficientNetB0(include_top=False, weights='imagenet')
        # self.base_model = VGG16(include_top=False, weights='imagenet')
        # Freeze the pretrained weights
        # self.base_model = tf.keras.Sequential([tf.keras.layers.GlobalAveragePooling2D(name="avg_pool"),
        #                                        tf.keras.layers.BatchNormalization(),
        #                                tf.keras.layers.Dropout(0.5, name="top_dropout")])
        if freeze:
            self.base_model.trainable = False

        self.top = tf.keras.Sequential([tf.keras.layers.GlobalAveragePooling2D(name="avg_pool"),
                                       tf.keras.layers.BatchNormalization(),
                                       tf.keras.layers.Dropout(0.8, name="top_dropout")])
        self.classifier = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")

    def call(self, inputs, training=True):
        x = self.base_model(inputs)
        x = self.top(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = YogaPose(num_classes=3, freeze=True)
    model.build(input_shape=(None, 224, 224, 3))
    print(model.summary())
