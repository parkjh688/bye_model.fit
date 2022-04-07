import tensorflow as tf
from tensorflow.keras.utils import Progbar
from model import YogaPose
from dataset import load_data
from loss import CustomAccuracy

from silence_tensorflow import silence_tensorflow
silence_tensorflow()


class Trainer:
    def __init__(self, model, epochs, batch):
        self.model = model
        self.epochs = epochs
        self.batch = batch

    def train(self, train_dataset, loss_fn, optimizer, train_metric, DATASET_SIZE):
        for epoch in range(self.epochs):
            print("\nStart of epoch %d" % (epoch,))
            # Iterate over the batches of the dataset.

            train_dataset = train_dataset.take(DATASET_SIZE//self.batch)

            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    logits = model(x_batch_train, training=True)
                    loss_value = loss_fn(y_batch_train, logits)
                    # tf.print(loss_value)
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                # Update training metric.
                train_metric.update_state(y_batch_train, logits)

                # Log every 3 batches.
                if step % 3 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %d samples" % ((step + 1) * self.batch))
                    print(train_metric.result().numpy())

                # Display metrics at the end of each epoch.
            train_acc = train_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))

            saved_model_path = "./models"
            tf.saved_model.save(model, saved_model_path)


if __name__ == '__main__':
    epoch = 7
    batch_size = 32

    model = YogaPose(num_classes=3)
    dataset, DATASET_SIZE = load_data(data_path='./dataset', label_path='label.txt', batch=batch_size)

    # train_size = int(0.8 * DATASET_SIZE)
    # val_size = int(0.2 * DATASET_SIZE)
    # print(train_size, val_size)
    # train_dataset = dataset.take(train_size)
    # remaining = dataset.skip(train_size)
    # val_dataset = remaining.take(val_size)
    print(DATASET_SIZE, batch_size)
    print(DATASET_SIZE//batch_size)
    loss_function = tf.keras.losses.CategoricalCrossentropy()
    # loss_function = CustomAccuracy()
    optimizer = tf.keras.optimizers.Adam()
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

    trainer = Trainer(model=model,
                      epochs=epoch,
                      batch=batch_size)
    trainer.train(train_dataset=dataset,
                  loss_fn=loss_function,
                  optimizer=optimizer,
                  train_metric=train_acc_metric,
                  DATASET_SIZE=DATASET_SIZE)
