# Based on the StarGAN paper and their PyTorch implementation
# https://github.com/yunjey/stargan
# @InProceedings{StarGAN2018,
# author = {Choi, Yunjey and Choi, Minje and Kim, Munyoung and Ha, Jung-Woo and Kim, Sunghun and Choo, Jaegul},
# title = {StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation},
# booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
# month = {June},
# year = {2018}
# }

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU, ReLU, Conv2D, UpSampling2D, Input, Reshape, Concatenate, ZeroPadding2D, Lambda
from tensorflow_addons.layers.normalizations import InstanceNormalization 
import numpy as np
import cv2
import glob
import re
import os
import random
from functools import partial
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

class Config(object):
	num_c = 7
	input_shape = (128,128,3)



# Model architecture as described in the StarGAN paper.

# Only need the generator to be build and loaded with weights for the project.
# Discriminator is kept for other testing.
class StarGAN():
  def __init__(self, config):
    self.input_shape = config.input_shape
    self.num_c = config.num_c
  
  def build_generator(self):  
    def conv2d(x, filters, kernel_size, strides, padding):
      x = ZeroPadding2D(padding=padding)(x)
      x = Conv2D(filters, kernel_size, strides, padding='valid', use_bias=False)(x)
      x = ReLU()(x)
      x = InstanceNormalization(axis=-1)(x)
      return x
    
    def deconv2d(x, filters, kernel_size, strides, padding):
      x = UpSampling2D(2)(x)
      x = Conv2D(filters, kernel_size, strides, padding='same', use_bias=False)(x)
      x = ReLU()(x)
      x = InstanceNormalization(axis=-1)(x)
      return x

    def down_sampling(x):
      d1 = conv2d(x, 64, 7, 1, 3)
      d2 = conv2d(d1, 128, 4, 2, 1)
      d3 = conv2d(d2, 256, 4, 2, 1)
      return d3

    def bottleneck(x):
      for _ in range(6):
        x = conv2d(x, 256, 3, 1, 1)
      return x
    
    def up_sampling(x):
      u1 = deconv2d(x, 128, 4, 1, 1)
      u2 = deconv2d(u1, 64, 4, 1, 1)
      return u2

    def output_conv(x):
      x = ZeroPadding2D(padding=3)(x)
      x = Conv2D(filters=3, kernel_size=7, strides=1, padding='valid', activation='tanh', use_bias=False)(x)
      return x
    
    input_img = Input(self.input_shape)
    input_c = Input((self.num_c,))
    c = Lambda(lambda x: backend.repeat(x, 128**2))(input_c)
    c = Reshape(self.input_shape)(c)
    x = Concatenate()([input_img, c])
    down_sampled = down_sampling(input_img)
    bottlenecked = bottleneck(down_sampled)
    up_sampled = up_sampling(bottlenecked)
    out = output_conv(up_sampled)
    return Model(inputs=[input_img, input_c], outputs=out)

  def build_discriminator(self):
    def conv2d(x, filters, kernel_size, strides, padding):
      x = ZeroPadding2D(padding=padding)(x)
      x = Conv2D(filters, kernel_size, strides, padding='valid', use_bias=False)(x)
      x = LeakyReLU(0.01)(x)
      return x
    
    input_img = Input(self.input_shape)
    x = input_img
    filters = 64
    for _ in range(6):
      x = conv2d(x, filters, 4, 2, 1)
      filters = filters*2

    out_cls = Conv2D(self.num_c, 2, 1, padding='valid', use_bias=False)(x)
    out_cls = Reshape((self.num_c,))(out_cls)
    x = ZeroPadding2D(padding=1)(x)
    out_src = Conv2D(1, 3, 1, padding='valid', use_bias=False)(x)
    return Model(inputs=input_img, outputs=[out_src, out_cls])
  

