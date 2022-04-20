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


# if __name__ == '__main__':
#     inputs = {'input_1':tf.ones([1, 224, 224, 3]), 'input_2':tf.ones([1, 17, 2])}
#     model = YogaPose(num_classes=3, freeze=True)
#     print(model(inputs))
#     # model.build(input_shape=(None, 224, 224, 3))
#     print(model.summary())
#     dot_img_file = './model_1.png'
#     tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)



# from weakref import finalize
# import tensorflow as tf
# from tensorflow.keras.utils import Progbar
# from model import YogaPose
# from dataset3 import load_data, show_batch
# from loss import CustomAccuracy
# import math

# import ipykernel
# import log
# logger = log.setup_custom_logger('root')

# # import wandb
# # wandb.init()

# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()


# class Trainer:
#     def __init__(self, model, epochs, batch):
#         self.model = model
#         self.epochs = epochs
#         self.batch = batch

#     # @tf.function
#     # def train_step(x, y):
#     #     with tf.GradientTape() as tape:
#     #         logits = model(x, training=True)
#     #         loss_value = loss_fn(y, logits)
#     #     grads = tape.gradient(loss_value, model.trainable_weights)
#     #     optimizer.apply_gradients(zip(grads, model.trainable_weights))
#     #     train_acc_metric.update_state(y, logits)
#     #     return loss_value

#     @tf.function
#     def compute_acc(self, y_pred, y):
#         correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
#         accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
#         return accuracy

#     def train_summary(self, ):
#         pass

#     def train(self, train_dataset, train_metric, steps_per_epoch, val_dataset, loss_fn, optimizer, val_metric, val_step, checkpoint_manager):
#         metrics_names = ['train_loss', 'val_loss']

#         # status = checkpoint.restore(manager.latest_checkpoint)

#         best_loss = 100
#         for epoch in range(self.epochs):
#             print("\nEpoch {}/{}".format(epoch+1, self.epochs))

#             train_dataset = train_dataset.shuffle(100)
#             val_dataset = val_dataset.shuffle(100)

#             train_dataset = train_dataset.take(steps_per_epoch)
#             val_dataset = val_dataset.take(val_step)

#             # for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
#             #     # print(step, y_batch_train.shape)
#             # train_dataset = train_dataset.take(DATASET_SIZE // self.batch)
#             # val_dataset = val_dataset.take(DATASET_SIZE // self.batch)

#             progBar = Progbar((steps_per_epoch + val_step) * self.batch, stateful_metrics=metrics_names)
#             train_loss = 0.5
#             val_loss = 0.5
#             # 데이터 집합의 배치에 대해 반복합니다
#             for step_train, (x_batch_train, y_batch_train) in enumerate(train_dataset):
#                 with tf.GradientTape() as tape:

#                     # show_batch(x_batch_train, y_batch_train,class_names=['adho mukha vriksasana', 'agnistambhasana', 'anjaneyasana'])
#                     logits = model(x_batch_train, training=True)    # 모델이 예측한 결과

#                     # print(y_batch_train, logits)

#                     train_loss = loss_fn(y_batch_train, logits)     # 모델이 예측한 결과와 GT를 이용한 loss 계산
#                 grads = tape.gradient(train_loss, model.trainable_weights)  # gradient 계산
#                 optimizer.apply_gradients(zip(grads, model.trainable_weights))  # Otimizer에게 처리된 그라데이션 적용을 요청
#                 # train metric(mean, auc, accuracy 등) 업데이트
#                 train_metric.update_state(y_batch_train, logits)

#                 train_acc = self.compute_acc(logits, y_batch_train)
#                 values = [('train_loss', train_loss), ('train_acc', train_acc)]
#                 # wandb.log({"train_loss": train_loss.numpy(), "train_acc": train_acc.numpy()})
#                 progBar.update(step_train * self.batch, values=values)

#             # progBar = Progbar(val_step * self.batch, stateful_metrics=metrics_names)
#             for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
#                 logits = model(x_batch_val, training=False)
#                 val_loss = loss_fn(y_batch_val, logits)
#                 # Update val metrics
#                 val_metric.update_state(y_batch_val, logits)
#                 val_acc = self.compute_acc(logits, y_batch_val)

#                 values = [('train_loss', train_loss), ('train_acc', train_acc), ('val_loss', val_loss), ('val_acc', val_acc)]
#                 progBar.update((step+step_train) * self.batch, values=values, finalize=True)

#                 val_metric.reset_states()

#                 # # 200 배치마다 로그를 남김
#                 # if step % 1 == 0:
#                 #     print(
#                 #         "Training loss (for one batch) at step %d: %.4f"
#                 #         % (step, float(loss_value))
#                 #     )
#                 #     print("Seen so far: %d samples" % ((step + 1) * self.batch))
#             if val_loss < best_loss:
#                 best_loss = val_loss
#                 manager.save()
#             # 매 epoch이 끝날 때 마다 train metrics 결과를 보여줌
#             train_acc = train_metric.result()
#             # print("Training acc over epoch: %.4f" % (float(train_acc),))
#             # train_metric.reset_states()
#             # saved_model_path = "./models"
#             # tf.saved_model.save(model, saved_model_path)


# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     num_classes = 3
#     epoch = 30
#     batch_size = 64

#     model = YogaPose(num_classes=num_classes)

#     test_path = '/Users/edensuperb//Downloads/pythonProject/dataset'
#     # val_path = '/Users/edensuperb//Downloads/pythonProject/dataset'

#     test, TEST_SIZE = load_data(data_path=test_path, img_shape=(224, 224), batch_size=32)

#     # logger.info('Test Data Size : {}'.format(TRAIN_SIZE))

#     # compute_steps_per_epoch = lambda x: int(math.ceil(1. * x / batch_size))
#     # steps_per_epoch = compute_steps_per_epoch(TRAIN_SIZE)
#     # val_steps = compute_steps_per_epoch(VAL_SIZE)

#     # print(steps_per_epoch, val_steps)

#     loss_function = tf.keras.losses.CategoricalCrossentropy()
#     # loss_function = CustomAccuracy()
#     optimizer = tf.keras.optimizers.Adam()
#     train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
#     val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

#     save_path = '/Users/edensuperb/bye_model.fit/ckpt-13'
#     checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
#     checkpoint.restore(save_path)

#     import numpy as np
#     for step_train, (x_batch_train, y_batch_train) in enumerate(test.take(10)):
#         # print(model(x_batch_train))
#         prediction = model(x_batch_train)
#         # print(tf.argmax(y_batch_train, axis=1))
#         # print(tf.argmax(prediction, axis=1))
#         # print(tf.equal(tf.argmax(y_batch_train, axis=1), tf.argmax(prediction, axis=1)))
#         print(np.array(tf.equal(tf.argmax(y_batch_train, axis=1), tf.argmax(prediction, axis=1))).sum())
#         # print("Prediction: {}".format(tf.argmax(prediction, axis=1)))

#     # trainer = Trainer(model=model,
#     #                   epochs=epoch,
#     #                   batch=batch_size)

#     # trainer.train(train_dataset=train_ds,
#     #               steps_per_epoch=steps_per_epoch,
#     #               val_step=val_steps,
#     #               val_dataset=val_ds,
#     #               loss_fn=loss_function,
#     #               optimizer=optimizer,
#     #               train_metric=train_acc_metric,
#     #               val_metric=val_acc_metric,
#     #               checkpoint_manager=manager)
