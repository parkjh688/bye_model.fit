import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import VGG16

class YogaPose(tf.keras.Model):
    def __init__(self, num_classes=30, freeze=False):
        super(YogaPose, self).__init__()
        # self.base_model = EfficientNetB0(include_top=False, weights='imagenet')
        self.keypoint = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(17, 2)),
                                             tf.keras.layers.Dense(34),])
                                            #  tf.keras.layers.BatchNormalization(),
                                            #  tf.keras.layers.Dropout(0.6, name="dropout")])
        # self.base_model = VGG16(include_top=False, weights='imagenet')
        # Freeze the pretrained weights
        # self.base_model = tf.keras.Sequential([tf.keras.layers.GlobalAveragePooling2D(name="avg_pool"),
        #                                        tf.keras.layers.BatchNormalization(),
        #                                tf.keras.layers.Dropout(0.5, name="top_dropout")])
        
        self.base_model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(224, 224, 3)),
                                            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
                                            tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation="relu"),
                                            tf.keras.layers.MaxPooling2D(pool_size=(2,2))
                                            ])

        if freeze:
            self.base_model.trainable = False

        self.top = tf.keras.Sequential([tf.keras.layers.GlobalAveragePooling2D(name="avg_pool"),
                                       tf.keras.layers.BatchNormalization(),
                                       tf.keras.layers.Dropout(0.6, name="top_dropout")])
        
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.classifier = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")

    def call(self, inputs, training=True):
        image, keypoint = inputs['input_1'], inputs['input_2']
        x1 = self.base_model(image)
        x1 = self.top(x1)
        x2 = self.keypoint(keypoint)
        x = self.concat([x1, x2])
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    inputs = {'input_1':tf.ones([1, 224, 224, 3]), 'input_2':tf.ones([1, 17, 2])}
    model = YogaPose(num_classes=3, freeze=True)
    print(model(inputs))
    # model.build(input_shape=(None, 224, 224, 3))
    print(model.summary())
    dot_img_file = './model_1.png'
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
