import tensorflow as tf

from model import YogaPose
from dataset3 import load_data

import numpy as np

import argparse

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch",          type=int,       default=100)
    parser.add_argument("--num_classes",    type=int,       default=3)
    parser.add_argument("--img_size",       type=int,       default=224)
    parser.add_argument("--test_path",      type=str,       default='./dataset/test')
    parser.add_argument("--checkpoint_path",type=str,		default='./checkpoints/ckpt-77')

    args = parser.parse_args()

    model = YogaPose(num_classes=args.num_classes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    test, TEST_SIZE = load_data(data_path=args.test_path, img_shape=(224, 224), batch_size=32)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint.restore(args.checkpoint_path)

    for step_train, (x_batch_train, y_batch_train) in enumerate(test.take(10)):
        # print(model(x_batch_train))
        prediction = model(x_batch_train)
        # print(tf.argmax(y_batch_train, axis=1))
        # print(tf.argmax(prediction, axis=1))
        # print(tf.equal(tf.argmax(y_batch_train, axis=1), tf.argmax(prediction, axis=1)))
        print("{}/{}".format(np.array(tf.equal(tf.argmax(y_batch_train, axis=1), tf.argmax(prediction, axis=1))).sum(), tf.argmax(y_batch_train, axis=1).shape[0]))
        # print("Prediction: {}".format(tf.argmax(prediction, axis=1)))


