import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer


class ResNet(Model):
    def __init__(self):
        super(ResNet, self).__init__(trainable=True)
        self.backbone = tf.keras.applications.ResNet50(include_top=False, input_shape=[96, 96, 3])
        self.final_conv = tf.keras.layers.Conv2D(filters=60, kernel_size=[3, 3], padding='valid')
        self.final_flatten = tf.keras.layers.Flatten()

    def call(self, inputs, training=None, mask=None):
        x = self.backbone(inputs)
        x = self.final_conv(x)
        x = self.final_flatten(x)
        return x, tf.keras.activations.softmax(x)


class EfficientNetB3(Model):
    def __init__(self):
        super(EfficientNetB3, self).__init__(trainable=True)
        self.backbone = tf.keras.applications.EfficientNetB3(include_top=False, input_shape=[150, 150, 3])
        self.final_conv = tf.keras.layers.Conv2D(filters=60, kernel_size=[3, 3], padding='valid')
        self.final_flatten = tf.keras.layers.Flatten()

    def call(self, inputs, training=None, mask=None):
        x = self.backbone(inputs)
        x = self.final_conv(x)
        x = self.final_flatten(x)
        return x, tf.keras.activations.softmax(x)
