import os
import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

    
def get_label(file_path, class_names):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2] == class_names


def process_path(file_path, class_names, img_shape=(224, 224)):
    label = tf.strings.split(file_path, os.path.sep)
    label = label[-2] == class_names

    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, img_shape)
    return img, label


def prepare_for_training(ds, batch_size=32, cache=True, shuffle_buffer_size=1000):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


def load_label(label_path):
    class_names = []
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            class_names.append(line)

    return np.array(class_names)


def show_batch(image_batch, label_batch, class_names):
    size = len(image_batch)
    sub_size = int(size ** 0.5) + 1

    plt.figure(figsize=(10, 10), dpi=80)
    for n in range(size):
        plt.subplot(sub_size, sub_size, n+1)
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)
        plt.title(class_names[label_batch[n]==True][0].title())
        plt.imshow(image_batch[n])
    plt.show()


def load_data(data_path, label_path, batch=32):
    class_names = load_label(label_path)
    data_dir = pathlib.Path(data_path)
    list_ds = tf.data.Dataset.list_files(str(data_dir / '*/*'))

	# 데이터 확인
    # for f in list_ds.take(5):
    #     print(f.numpy())

    labeled_ds = list_ds.map(lambda x: process_path(x, class_names))
    train_ds = prepare_for_training(labeled_ds, batch_size=batch)

    return train_ds