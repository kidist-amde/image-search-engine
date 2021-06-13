import tensorflow as tf
import random
import numpy as np
import wandb


class ImageClassifier():
    def __init__(self, batch_size, epochs, ilr=0.0001):
        self.epochs = epochs
        self.batch_size = batch_size
        self.initial_learning_rate = ilr

    def train(self, training_data, training_labels, test_data, test_labels, model):
        # specify loss, SparseCategoricalCrossentropy is good in image classification
        supervised_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        # logits = output of neural network before the final activation function, thus x in call()
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.initial_learning_rate,
            decay_steps=500,
            decay_rate=0.96,
            staircase=True)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule)  # adam = adaptive stochastic gradient descent

        @tf.function
        def train_step(data, labels):
            with tf.GradientTape() as tape:
                logits, preds = model(data, training=True)
                loss = supervised_loss(y_true=labels, y_pred=logits)
            trainable_vars = model.trainable_variables  # get the weights
            gradients = tape.gradient(loss, trainable_vars)  # get the gradients
            optimizer.apply_gradients(
                zip(gradients, trainable_vars))  # apply the gradients to the weights to update them
            # I have optimized it, now compute the accuracy
            eq = tf.equal(labels, tf.argmax(preds,
                                            -1))  # tf.argmax get the class with maximum value (-1 is the axis, thus columns where there are stored the prob of images to be of a class)
            accuracy = tf.reduce_mean(tf.cast(eq, tf.float32)) * 100
            return loss, accuracy

        @tf.function
        def test_step(data, labels):
            logits, preds = model(data, training=False)
            loss = supervised_loss(y_true=labels, y_pred=logits)
            return loss, logits, preds

        global_step = 0
        best_accuracy = 0.0
        for e in range(self.epochs):  # for over the epochs, how many times I run the train

            ## Shuffling training data at every epoch
            # to not let the model see the data in an order way, so the model see the most various type of inputs
            perm = np.arange(len(training_labels))
            random.shuffle(perm)
            training_data = training_data[perm]
            training_labels = training_labels[perm]

            ## Iteration
            for i in range(0, len(training_labels), self.batch_size):  # step of batch size, so i will be 0, 31, ..
                data = training_data[i:i + self.batch_size, :]
                labels = training_labels[i:i + self.batch_size, ].astype('int64')
                global_step += 1
                batch_loss, batch_accuracy = train_step(data, labels)
                if global_step % 50 == 0:  # print every 50 steps
                    print('[{0}-{1:03}] loss: {2:0.05}, batch_accuracy: {3:0.03}'.format(
                        e + 1, global_step,
                        batch_loss.numpy(),
                        batch_accuracy.numpy()))
                    # to log informations to wandb create a dictionary and log them
                    lr = optimizer._decayed_lr(tf.float32).numpy()
                    wandb.log(
                        {'train/batch_loss': batch_loss,
                         'train/batch_accuracy': batch_accuracy,
                         'train/learning_rate': lr
                         })
                if global_step == 1:
                    print('number of model parameters {}'.format(model.count_params()))

            # When epoch is finished run the test
            test_preds = tf.zeros((0,), dtype=tf.int64)  # array that will contain the test predictions
            total_loss = list()
            for i in range(0, len(test_labels), self.batch_size):
                data = test_data[i:i + self.batch_size, :]
                labels = test_labels[i:i + self.batch_size, ].astype('int64')
                batch_loss, _, preds = test_step(data, labels)
                batch_preds = tf.argmax(preds, -1)  # get class predicted of the batch
                test_preds = tf.concat([test_preds, batch_preds], axis=0)
                total_loss.append(batch_loss)
            loss = sum(total_loss) / len(total_loss)
            eq = tf.equal(test_labels, test_preds)
            test_accuracy = tf.reduce_mean(tf.cast(eq, tf.float32)) * 100
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                model.save('./mymodel4')
            print('End of Epoch {0}/{1:03} -> loss: {2:0.05}, test accuracy: {3:0.03} - best accuracy: {4:0.03}'.format(
                e + 1, self.epochs,
                loss.numpy(),
                test_accuracy.numpy(),
                best_accuracy))
            # to log informations to wandb create a dictionary and log them
            wandb.log(
                {'test/loss': loss,
                 'test/accuracy': test_accuracy,
                 'test/best_accuracy': best_accuracy,
                 'epoch': e + 1})



