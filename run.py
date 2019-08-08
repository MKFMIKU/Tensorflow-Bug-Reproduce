from os import listdir
from os.path import join

import tensorflow as tf
from tensorflow import keras
import net_ops as ops


def build_model():
    grid_th = tf.keras.Input(shape=(1, 16, 16, 8, 12))
    guide_th = tf.keras.Input(shape=(None, None, 1))
    image_th = tf.keras.Input(shape=(None, None, 3))
    grid_th = tf.keras.activations.relu(grid_th)
    output_th = ops.bilateral_slice_apply(grid_th, guide_th, image_th, has_offset=True)
    model = tf.keras.Model(inputs=(grid_th, guide_th, image_th), outputs=output_th)
    return model


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 1

model = build_model()

grid_th = tf.random.normal(shape=[1, 16, 16, 8, 12])
guide_th = tf.random.normal(shape=[256, 256, 1])
image_th = tf.random.normal(shape=[256, 256, 3])
with tf.GradientTape() as tape:
    output = model(grid_th, guide_th, image_th)
    loss = tf.keras.losses.mse(output, image_th)
grads = tape.gradient(loss, model.trainable_variables)
print(grads)
