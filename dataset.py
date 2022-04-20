import os
import pathlib

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import logging
import json

logger = logging.getLogger('root')

with open("/Users/edensuperb/Downloads/keypoints.json", "r") as json_file:
  keypoint_dict = json.load(json_file)

new_keypoint_dict = {}
for key in keypoint_dict.keys():
    new_key = '/'.join(key.split('/')[-2:])
    new_val = keypoint_dict[key]
    new_keypoint_dict[new_key] = new_val

del keypoint_dict
keypoint_dict = new_keypoint_dict
del new_keypoint_dict
    

# print(len(keypoint_dict.keys()))
# for i in keypoint_dict.keys():
#     print(np.array(keypoint_dict[i]).shape)
# print(type(keypoint_dict))

def process_keypoint(file_path):
    file_path =  '/'.join(file_path.numpy().decode('utf-8').split('/')[-2:])
    # keypoint = np.array(keypoint_dict[file_path])
    keypoint = tf.convert_to_tensor(keypoint_dict[file_path], dtype=tf.float32)
    return keypoint

def process_path(file_path, class_names, img_shape=(224, 224)):
    label = tf.strings.split(file_path, os.path.sep)
    label = label[-2] == class_names
    label = tf.cast(label, tf.float32)

    img = tf.io.read_file(file_path)
    # img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, img_shape) 
    # img = img / 255.0

    [keypoint,] = tf.py_function(process_keypoint, [file_path], [tf.float32])
    # file_path.numpy().decode('utf-8').split('/')
    # file_path = '/'.join(file_path.split('/')[-2:])
    # keypoint = np.array(keypoint_dict[file_path])

    # return img, label
    return {"input_1": img, "input_2": keypoint}, label


def show_batch(image_batch, label_batch, class_names):
    size = len(image_batch)
    sub_size = int(size ** 0.5) + 1

    # plt.figure(figsize=(10, 10), dpi=80)
    # for n in range(size):
    #     plt.subplot(sub_size, sub_size, n+1)
    #     plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)
    #     plt.title(class_names[label_batch[n]==True][0].title())
    #     plt.imshow(image_batch[n])
    # plt.show()
    plt.figure(figsize=(10, 10), dpi=80)
    for n in range(size):
        plt.rc('font', size=10)
        plt.subplot(sub_size, sub_size, n+1)
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)
        # plt.title(class_names[label_batch[n]==True][0].title())
        idx = tf.where(label_batch[n]).numpy()[0][0]
        plt.title(class_names[idx])
        plt.imshow(image_batch[n])
    plt.show()


def load_label(label_path):
    class_names = []
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            class_names.append(line)

    return np.array(class_names)


def get_spilt_data(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=1000):
    assert (train_split + val_split) == 1

    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    # train_ds = train_ds.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = ds.skip(train_size).take(val_size)

    return train_ds, val_ds


def augment(inputs, label):
    image, keypoint = inputs['input_1'], inputs['input_2']
    # image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]
    # image = tf.image.resize_with_crop_or_pad(image, 34, 34) # Add 6 pixels of padding
    image = tf.image.random_crop(image, size=[224, 224, 3])
    # image = tf.image.flip_left_right(image)
    # image = tf.image.random_flip_up_down(image)
    image = tf.image.adjust_brightness(image, 0.4)
    image = tf.image.random_brightness(image, max_delta=0.4) # Random brightness
    return {'input_1' : image, 'input_2' : keypoint}, label


def prepare_for_training(ds, batch_size=32, cache=True, training=True):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    # ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()
    # if training:
    #     ds = ds.map(lambda x, y: augment(x, y))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


def load_data(data_path, img_shape, batch_size=32, is_train=True):
    class_names = [cls for cls in os.listdir(data_path) if cls != '.DS_Store']
    data_dir = pathlib.Path(data_path)
    list_ds = tf.data.Dataset.list_files(str(data_dir / '*/*'))

    # logger.info('Find {} class'.format(class_names))

    labeled_ds = list_ds.map(lambda x: process_path(x, class_names, img_shape))
    labeled_ds = prepare_for_training(labeled_ds, batch_size=batch_size, training=is_train)

    DATASET_SIZE = tf.data.experimental.cardinality(list_ds).numpy()

    return labeled_ds, DATASET_SIZE


if __name__ == '__main__':
    train_path = '/Users/edensuperb//Downloads/dataset'

    train_ds, train_size = load_data(data_path=train_path, img_shape=(224, 224), batch_size=1)

    for inputs, label in train_ds.take(1):
        # image, keypoint = inputs['input_1'], inputs['input_2']
        print(label)
        # show_batch(img, label, class_names)