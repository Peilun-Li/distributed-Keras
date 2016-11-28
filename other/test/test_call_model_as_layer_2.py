import tensorflow as tf
from keras.metrics import categorical_accuracy as accuracy
from keras.layers import Dropout
from keras import backend as K
from tensorflow.examples.tutorials.mnist import input_data
from keras.objectives import categorical_crossentropy
from keras.layers import *
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model

sess = tf.Session()
K.set_session(sess)

mnist_data = input_data.read_data_sets('../MNIST_data', one_hot=True)

img1 = tf.placeholder(tf.float32, shape=(None, 784))
img2 = tf.placeholder(tf.float32, shape=(None, 784))
labels = tf.placeholder(tf.float32, shape=(None, 10))

inputs1 = Input(shape=(784,))
inputs2 = Input(shape=(784,))
x = merge([inputs1, inputs2], mode='concat')
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
pred = Dense(10, activation='softmax')(x)

model = Model(input=[inputs1,inputs2], output=pred)

model.save('test_call_model_as_layer.h5')
del model
model = load_model('test_call_model_as_layer.h5')

preds = model([img1, img2])


loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
with sess.as_default():
    for i in range(100):
        batch = mnist_data.train.next_batch(50)
        #train_step.run(feed_dict={img1: batch[0],
        #                          img2: batch[0],
        #                          labels: batch[1],
        #                          K.learning_phase(): 1})
        _, cur_loss = sess.run([train_step, loss], feed_dict={img1: batch[0],img2: batch[0],labels: batch[1],K.learning_phase(): 1})
        print cur_loss

acc_value = accuracy(labels, preds)
with sess.as_default():
    print acc_value.eval(feed_dict={img1: mnist_data.test.images,
                                    img2: mnist_data.test.images,
                                    labels: mnist_data.test.labels,
                                    K.learning_phase(): 0})