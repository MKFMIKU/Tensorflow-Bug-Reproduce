from os import listdir
from os.path import join

import tensorflow as tf
from tensorflow import keras

import net_ops as ops
def bilateral_slice_apply(input):
    grid, guide, input_image = input
    sliced = ops.bilateral_slice_apply(grid, guide, input_image, has_offset=True)
    return sliced


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__(name='model')

    def call(self, grid_th, guide_th, image_th):
        image_th = tf.keras.layers.Conv2D(3, 3, padding='same')(image_th) 
        output_th = ops.bilateral_slice_apply(grid_th, guide_th, image_th, has_offset=True)
        return output_th


model = Model()
grid_th = tf.random.normal(shape=[1, 16, 16, 8, 12])
guide_th = tf.random.normal(shape=[1, 256, 256])
image_th = tf.random.normal(shape=[1, 256, 256, 3])
label_th = tf.random.normal(shape=[1, 256, 256, 3])
with tf.GradientTape() as tape:
    output = model(grid_th, guide_th, image_th)
    loss = tf.keras.losses.mse(output, label_th)
grads = tape.gradient(loss, model.trainable_variables)
print(model.trainable_variables)
print(loss, grads)
