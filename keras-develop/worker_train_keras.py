import tensorflow as tf
import image_processing
from keras.models import *
from keras.layers import *
from keras import backend as K
from keras.objectives import *
from keras.metrics import *
import numpy as np
import sys
import ast
import os
from datetime import datetime
import socket
import scipy.misc as smp
'''
Train model on each worker
'''
def get_time():
    return socket.gethostbyname(socket.gethostname()) + " " + datetime.now().strftime('%H:%M:%S')

def worker_train(model_path, data_dir, batch_size, preprocess_operation,
                 image_size, num_classes, train_steps, val_stpes, val_interval,
                 objectives, do_evaluation = False, labels_offset = 0, num_preprocess_threads=None, num_readers=1,
                 examples_per_shard=1024, input_queue_memory_factor = 16):


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    model = load_model(model_path)
    print("loading tensorflow data reading queues...")
    train_image_tensor, train_labels_tensor, train_labels_text_tensor = image_processing.batch_inputs(
        data_dir, batch_size, image_size, True, preprocess_operation, num_preprocess_threads,
        num_readers, examples_per_shard, input_queue_memory_factor)

    val_image_tensor, val_labels_tensor, test_labels_text_tensor = image_processing.batch_inputs(
        data_dir, batch_size, image_size, False, preprocess_operation, num_preprocess_threads,
        num_readers, examples_per_shard, input_queue_memory_factor)
    tf.train.start_queue_runners(sess=sess)

    train_labels = train_labels_tensor
    val_labels = val_labels_tensor
    if labels_offset != 0:
        train_labels = tf.sub(train_labels, labels_offset)
        val_labels = tf.sub(val_labels, labels_offset)

    # check objectives
    if objectives == 'categorical_crossentropy':
        train_labels = tf.one_hot(train_labels, num_classes, axis=-1)
        train_labels = tf.cast(train_labels, tf.float32)
        val_labels = tf.one_hot(val_labels, num_classes, axis=-1)
        val_labels = tf.cast(val_labels, tf.float32)
    elif objectives == 'binary_crossentropy':
        train_labels = tf.cast(train_labels, tf.float32)
        train_labels = tf.reshape(train_labels, [-1,1])
        val_labels = tf.cast(val_labels, tf.float32)
        val_labels = tf.reshape(val_labels, [-1, 1])
    else:
        print('No corresponding objectives %s' % objectives)
        exit(-1)
    print("training...")

    with sess.as_default():
        for i in range(1, train_steps+1):
            train_x, train_y = sess.run([train_image_tensor, train_labels])
            print get_time(), model.train_on_batch(train_x, train_y)
            if i % val_interval == 0 and do_evaluation:
                val_res = []
                for j in range(val_stpes):
                    val_x, val_y = sess.run([val_image_tensor, val_labels])
                    val_res.append(model.test_on_batch(val_x, val_y))
                print get_time(), "val: ", np.mean(val_res, 0).tolist()

    cur_dir_path = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(cur_dir_path, 'trained_model.h5')
    model.save(save_path)

if __name__ == '__main__':
    '''
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(150, 150, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.save('test_call_model_as_layer.h5')
    del model
    '''
    args = sys.argv
    if len(args) > 1:
        worker_train(args[1], args[2],
                     int(args[3]), args[4], ast.literal_eval(args[5]), int(args[6]), int(args[7]), int(args[8]), int(args[9]),
                    args[10], args[11] == 'True', int(args[12]),
                     int(args[13]), int(args[14]), int(args[15]), int(args[16]))
    else:

        worker_train('predefined_model.h5', 'data/tfrecords/0',
                 100, 'resize', [150, 150, 3], 2, 300, 10, 20,
                 'categorical_crossentropy', do_evaluation=True, labels_offset=1,
                 num_preprocess_threads=4, num_readers=1, examples_per_shard=100, input_queue_memory_factor=2)

        '''
        worker_train('resnet50.h5', '/home/lpl/Documents/dataset/ILSVRC2015_subsample/tfrecords/0',
                     1, 'resize', [224, 224, 3], 1000, 300, 10, 20,
                     'categorical_crossentropy', do_evaluation=True, labels_offset=1,
                     num_preprocess_threads=4, num_readers=1, examples_per_shard=100, input_queue_memory_factor=2)
        '''
        '''
        worker_train('inception_v3.h5', '/home/lpl/Documents/dataset/ILSVRC2015_subsample/tfrecords/0',
                     1, 'resize', [299, 299, 3], 1000, 300, 10, 20,
                     'categorical_crossentropy', do_evaluation=True, labels_offset=1,
                     num_preprocess_threads=4, num_readers=1, examples_per_shard=100, input_queue_memory_factor=2)
        '''