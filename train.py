import keras
import numpy as np
import tensorflow as tf
from keras.applications import resnet50
#Load the ResNet50 model

from trax import math
from keras import backend as K

import tensorflow_datasets as tfds

from trax.models.transformer import TransformerEncoder
from trax import layers as tl
from trax.shapes import ShapeDtype

from keras.models import Model
from keras.layers import Input, Dense

IMAGE_WIDTH = IMAGE_HEIGHT = 224

def _resize_sample(image):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
  return image

ds_train = tfds.load("imagenet2012", split=tfds.Split.TRAIN, as_supervised=True)
ds_val = tfds.load("imagenet2012", split=tfds.Split.VALIDATION, as_supervised=True)


def preprocess(ds):
  ds = ds.map(lambda x, y: (_resize_sample(x), y))
  ds = ds.map(lambda x, y: (resnet50.preprocess_input(x), y))
  ds = ds.shuffle(1000)
  ds = ds.batch(50)
  ds = ds.prefetch(2)
  return ds

resnet_model = resnet50.ResNet50(include_top=True, weights='imagenet')

inp = resnet_model.input

activations = []
layers = resnet_model.layers

for layer in layers:
  if isinstance(layer, keras.layers.core.Activation):
    activations.append(tf.squeeze(tf.nn.max_pool2d(layer.output, layer.output.get_shape()[1:3], 1, 'VALID')))


activations = activations[-1:]
functor = K.function([inp, K.learning_phase()], activations)


vocab_size = 2048


te = TransformerEncoder(vocab_size,
                       n_classes=1000,
                       d_model=512,
                       d_ff=2048,
                       n_layers=6,
                       n_heads=8,
                       dropout=0.1,
                       max_len=2048,
                       mode='train',
                       ff_activation=tl.Relu)

rng = math.random.get_prng(0)

batch_size = 1
num_classes = 1000

input_signature = ShapeDtype((batch_size, 5), np.int32)

te.init(input_signature)

feature_tokens = np.array([[0,1,2,3,4]])
logp = te(feature_tokens, rng=rng)
p = tf.exp(logp)

target_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size, num_classes])
loss = tf.square(p-target_ph)

sess = tf.compat.v1.keras.backend.get_session()
optimizer = tf.train.AdamOptimizer(learning_rate=0.0003)
grads = optimizer.compute_gradients(loss)
apply_grads_op = optimizer.apply_gradients(grads)


ds_train = preprocess(ds_train)

for data in ds_train:
  x, y = data
  y_pred = resnet_model.predict(x, steps=1)
  y_pred = np.argmax(y_pred, axis=-1)
  acc = np.count_nonzero(y_pred == y)
  print(acc)
  feed_dict = { target_ph: y }
  sess.run([apply_grads_op], feed_dict=feed_dict)
  #outs = functor([x, 0])
  #print([out.shape for out in outs])






from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions


filename = 'images/cat.jpg'
original_image = load_img(filename, target_size=(224, 224))
numpy_image = img_to_array(original_image)
input_image = np.expand_dims(numpy_image, axis=0)

processed_image_resnet50 = resnet50.preprocess_input(input_image.copy())


