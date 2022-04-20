import tensorflow as tf
from tensorflow.keras.utils import Progbar

import math
import numpy as np

from model import YogaPose
from dataset import load_data


import log
logger = log.setup_custom_logger('root')

# import wandb
# wandb.init()

from silence_tensorflow import silence_tensorflow
silence_tensorflow()


class Trainer:
    def __init__(self, model, epochs, batch, loss_fn, optimizer):
        self.model = model
        self.epochs = epochs
        self.batch = batch
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def compute_acc(self, y_pred, y):
        correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        return accuracy

    @tf.function
    def train_on_batch(self, x_batch_train, y_batch_train):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)    # 모델이 예측한 결과
            train_loss = self.loss_fn(y_batch_train, logits)     # 모델이 예측한 결과와 GT를 이용한 loss 계산

        grads = tape.gradient(train_loss, model.trainable_weights)  # gradient 계산
        self.optimizer.apply_gradients(zip(grads, model.trainable_weights))  # Otimizer에게 처리된 그라데이션 적용을 요청

        return train_loss, logits

    def train(self, train_dataset, acc_metric, steps_per_epoch, val_dataset, val_step, checkpoint_manager):
        metrics_names = ['train_loss', 'train_acc', 'val_loss']

        # status = checkpoint.restore(manager.latest_checkpoint)

        best_loss = 100
        for epoch in range(self.epochs):
            print("\nEpoch {}/{}".format(epoch+1, self.epochs))

            train_dataset = train_dataset.shuffle(100)
            val_dataset = val_dataset.shuffle(100)

            train_dataset = train_dataset.take(steps_per_epoch)
            val_dataset = val_dataset.take(val_step)

            progBar = Progbar(steps_per_epoch * self.batch, stateful_metrics=metrics_names)

            train_loss, val_loss = 100, 100

            # 데이터 집합의 배치에 대해 반복합니다
            for step_train, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                train_loss, logits = self.train_on_batch(x_batch_train, y_batch_train)

                # train metric(mean, auc, accuracy 등) 업데이트
                acc_metric.update_state(y_batch_train, logits)

                train_acc = self.compute_acc(logits, y_batch_train)
                values = [('train_loss', train_loss), ('train_acc', train_acc)]
                # print('{}'.format((step_train + 1) * self.batch))
                progBar.update((step_train + 1) * self.batch, values=values)

            for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
                logits = model(x_batch_val, training=False)
                val_loss = self.loss_fn(y_batch_val, logits)
                val_acc = self.compute_acc(logits, y_batch_val)
                values = [('train_loss', train_loss), ('train_acc', train_acc), ('val_loss', val_loss), ('val_acc', val_acc)]
            progBar.update((step_train + 1) * self.batch, values=values, finalize=True)

            # wandb.log({'accuracy': train_acc, 'loss': train_loss})

            if val_loss < best_loss:
                best_loss = val_loss
                print("\nSave better model: ", end='')
                print(checkpoint_manager.save())
            # # 매 epoch이 끝날 때 마다 train metrics 결과를 보여줌
            # train_acc = acc_metric.result()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch",          type=int,       default=100)
    parser.add_argument("--batch_size",     type=int,       default=64)
    parser.add_argument("--num_classes",    type=int,       default=3)
    parser.add_argument("--img_size",       type=int,       default=224)
    parser.add_argument("--train_path",     type=str,		default='./dataset/train')
    parser.add_argument("--val_path",       type=str,		default='./dataset/val')
    parser.add_argument("--checkpoint_path",type=str,		default='./checkpoints')

    args = parser.parse_args()

    model = YogaPose(num_classes=args.num_classes)

    train_ds, TRAIN_SIZE = load_data(data_path=args.train_path, img_shape=(args.img_size, args.img_size), batch_size=args.batch_size)
    val_ds, VAL_SIZE = load_data(data_path=args.val_path, img_shape=(args.img_size, args.img_size), batch_size=args.batch_size, is_train=False)

    logger.info('Train Data Size : {}'.format(TRAIN_SIZE))
    logger.info('Validation Data Size : {}'.format(VAL_SIZE))

    compute_steps_per_epoch = lambda x: int(math.ceil(1. * x / args.batch_size))
    steps_per_epoch = compute_steps_per_epoch(TRAIN_SIZE)
    val_steps = compute_steps_per_epoch(VAL_SIZE)

    loss_function = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.3)
    # loss_function = CustomAccuracy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    acc_metric = tf.keras.metrics.CategoricalAccuracy()

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, directory=args.checkpoint_path, max_to_keep=5)

    trainer = Trainer(model=model,
                      epochs=args.epoch,
                      batch=args.batch_size,
                      loss_fn=loss_function,
                      optimizer=optimizer,)

    trainer.train(train_dataset=train_ds,
                steps_per_epoch=steps_per_epoch,
                val_step=val_steps,
                val_dataset=val_ds,
                acc_metric=acc_metric,
                checkpoint_manager=manager)

