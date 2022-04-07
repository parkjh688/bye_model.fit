import tensorflow as tf
from tensorflow.keras.utils import Progbar
from model import YogaPose
from dataset import load_data
from loss import CustomAccuracy
import math

from silence_tensorflow import silence_tensorflow
silence_tensorflow()


class Trainer:
    def __init__(self, model, epochs, batch):
        self.model = model
        self.epochs = epochs
        self.batch = batch

    # @tf.function
    # def train_step(x, y):
    #     with tf.GradientTape() as tape:
    #         logits = model(x, training=True)
    #         loss_value = loss_fn(y, logits)
    #     grads = tape.gradient(loss_value, model.trainable_weights)
    #     optimizer.apply_gradients(zip(grads, model.trainable_weights))
    #     train_acc_metric.update_state(y, logits)
    #     return loss_value

    @tf.function
    def compute_acc(self, y_pred, y):
        correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        return accuracy

    def train_summary(self, ):
        pass

    def train(self, train_dataset, train_metric, steps_per_epoch, val_dataset, loss_fn, optimizer, val_metric, val_step):
        metrics_names = ['train_loss', 'val_loss']

        for epoch in range(self.epochs):
            print("\nEpoch {}/{}".format(epoch+1, self.epochs))

            train_dataset = train_dataset.shuffle(100)
            val_dataset = val_dataset.shuffle(100)

            train_dataset = train_dataset.take(steps_per_epoch)
            val_dataset = val_dataset.take(val_step)

            # for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            #     # print(step, y_batch_train.shape)
            # train_dataset = train_dataset.take(DATASET_SIZE // self.batch)
            # val_dataset = val_dataset.take(DATASET_SIZE // self.batch)

            progBar = Progbar(steps_per_epoch * self.batch, stateful_metrics=metrics_names)
            train_loss = 0.5
            val_loss = 0.5
            # 데이터 집합의 배치에 대해 반복합니다
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    logits = model(x_batch_train, training=True)    # 모델이 예측한 결과
                    train_loss = loss_fn(y_batch_train, logits)     # 모델이 예측한 결과와 GT를 이용한 loss 계산
                grads = tape.gradient(train_loss, model.trainable_weights)  # gradient 계산
                optimizer.apply_gradients(zip(grads, model.trainable_weights))  # Otimizer에게 처리된 그라데이션 적용을 요청
                # train metric(mean, auc, accuracy 등) 업데이트
                train_metric.update_state(y_batch_train, logits)

                train_acc = self.compute_acc(logits, y_batch_train)
                values = [('train_loss', train_loss), ('train_acc', train_acc)]
                progBar.update(step * self.batch, values=values)

            # progBar = Progbar(DATASET_SIZE * self.batch, stateful_metrics=metrics_names)
            for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
                logits = model(x_batch_val, training=False)
                val_loss = loss_fn(y_batch_val, logits)
                # Update val metrics
                val_metric.update_state(y_batch_val, logits)
                val_acc = self.compute_acc(logits, y_batch_val)

                values = [('train_loss', train_loss), ('val_loss', val_loss), ('val_acc', val_acc)]
                progBar.update(steps_per_epoch * self.batch, values=values, finalize=True)

                val_metric.reset_states()

                # # 200 배치마다 로그를 남김
                # if step % 1 == 0:
                #     print(
                #         "Training loss (for one batch) at step %d: %.4f"
                #         % (step, float(loss_value))
                #     )
                #     print("Seen so far: %d samples" % ((step + 1) * self.batch))

            # 매 epoch이 끝날 때 마다 train metrics 결과를 보여줌
            train_acc = train_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))
            train_metric.reset_states()
            # saved_model_path = "./models"
            # tf.saved_model.save(model, saved_model_path)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    epoch = 10
    batch_size = 32

    model = YogaPose(num_classes=8)
    train_ds, val_ds, DATASET_SIZE = load_data(data_path='./dataset', label_path='label.txt',
                                               batch_size=batch_size, )

    train_size = int(0.7 * DATASET_SIZE)
    val_size = int(0.3 * DATASET_SIZE)
    print(train_size, val_size)

    compute_steps_per_epoch = lambda x: int(math.ceil(1. * x / batch_size))
    steps_per_epoch = compute_steps_per_epoch(train_size)
    val_steps = compute_steps_per_epoch(val_size)

    print(steps_per_epoch, val_steps)
    # train_dataset = dataset.take(train_size)
    # remaining = dataset.skip(train_size)
    # val_dataset = remaining.take(val_size)
    #
    loss_function = tf.keras.losses.CategoricalCrossentropy()
    # loss_function = CustomAccuracy()
    optimizer = tf.keras.optimizers.Adam()
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

    trainer = Trainer(model=model,
                      epochs=epoch,
                      batch=batch_size)
    trainer.train(train_dataset=train_ds,
                  steps_per_epoch=steps_per_epoch,
                  val_step=val_steps,
                  val_dataset=val_ds,
                  loss_fn=loss_function,
                  optimizer=optimizer,
                  train_metric=train_acc_metric,
                  val_metric=val_acc_metric,)
                  # DATASET_SIZE=DATASET_SIZE)
