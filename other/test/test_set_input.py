import tensorflow as tf
from keras.metrics import categorical_accuracy as accuracy
from keras.layers import Dropout
from keras import backend as K
from tensorflow.examples.tutorials.mnist import input_data
from keras.objectives import categorical_crossentropy
from keras.layers import Dense
from keras.models import Sequential


sess = tf.Session()
K.set_session(sess)

mnist_data = input_data.read_data_sets('../MNIST_data', one_hot=True)

img = tf.placeholder(tf.float32, shape=(None, 784))
labels = tf.placeholder(tf.float32, shape=(None, 10))

model = Sequential()
first_layer = Dense(128, activation='relu', input_dim=784)
first_layer.set_input(img)
model.add(first_layer)
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
preds = model.output

loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
with sess.as_default():
    for i in range(100):
        batch = mnist_data.train.next_batch(50)
        train_step.run(feed_dict={img: batch[0],
                                  labels: batch[1],
                                  K.learning_phase(): 1})

acc_value = accuracy(labels, preds)
with sess.as_default():
    print acc_value.eval(feed_dict={img: mnist_data.test.images,
                                    labels: mnist_data.test.labels,
                                    K.learning_phase(): 0})