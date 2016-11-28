import tensorflow as tf
import image_processing
from keras.models import *
from keras.layers import *
from keras import backend as K
from keras.objectives import *
from keras.metrics import *


def worker_train(model_path, data_dir, batch_size, preprocess_operation,
                 image_size, num_classes, max_steps, objectives, optimizers,
                 learning_rate=0.01, labels_offset = 0, num_preprocess_threads=None, num_readers=1,
                 examples_per_shard=1024, input_queue_memory_factor = 16):

    sess = tf.Session()
    K.set_session(sess)

    model = load_model(model_path)
    train_image_tensor, train_labels_tensor = image_processing.batch_inputs(
        data_dir, batch_size, image_size, True, preprocess_operation, num_preprocess_threads,
        num_readers, examples_per_shard, input_queue_memory_factor)

    #x = tf.placeholder(tf.float32, shape=(None, image_size[0], image_size[1], image_size[2]))
    y = tf.placeholder(tf.int32, shape=(batch_size,))
    #preds = model(x)
    x = model.input
    preds = model.output

    train_labels = y
    if labels_offset != 0:
        train_labels = tf.sub(y, labels_offset)

    # check objectives
    if objectives == 'categorical_crossentropy':
        train_labels = tf.one_hot(train_labels, num_classes, axis=-1)
        train_labels = tf.cast(train_labels, tf.float32)
        loss = tf.reduce_mean(categorical_crossentropy(train_labels, preds))
        # this loss is the same as:
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(preds, train_labels))
        acc_value = categorical_accuracy(train_labels, preds)
    elif objectives == 'binary_crossentropy':
        # train_labels: 1-d tensor (batch_size,)
        # preds: 2-d tensor (batch_size, 1)
        train_labels = tf.cast(train_labels, tf.float32)
        preds = tf.reshape(preds, train_labels.get_shape().as_list())
        # or reshape train_labels as:
        # train_labels = tf.reshape(train_labels, [-1,1])
        loss = tf.reduce_mean(binary_crossentropy(train_labels, preds))
        # this loss is the same as:
        # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(preds, train_labels))
        acc_value = binary_accuracy(train_labels, preds)
    else:
        print('No corresponding objectives %s' % objectives)
        exit(-1)

    print(preds.get_shape().as_list())
    print(train_labels.get_shape().as_list())

    # check optimizers
    if optimizers == 'RMSProp':
        # recommend learning rate: around 0.001
        train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    elif optimizers == 'GradientDescent':
        # recommend learning rate: around 0.01
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    else:
        print('No corresponding optimizers %s' % optimizers)
        exit(-1)

    sess.run(tf.initialize_all_variables())
    tf.train.start_queue_runners(sess=sess)
    with sess.as_default():
        for i in range(max_steps):
            tx, ty = sess.run([train_image_tensor, train_labels_tensor])
            _, cur_loss, cur_accu= sess.run([train_step, loss, acc_value],
                                            feed_dict={ K.learning_phase(): 1, x:tx, y:ty})
            print cur_loss, cur_accu

    with open('json_model.json','w') as f:
        f.write(model.to_json())
    model.save_weights('json_model_weights.h5')


def worker_test(data_dir, batch_size, preprocess_operation,
                 image_size, num_classes, max_steps, objectives, optimizers,
                 learning_rate=0.01, labels_offset = 0, num_preprocess_threads=None, num_readers=1,
                 examples_per_shard=1024, input_queue_memory_factor = 16):

    sess = tf.Session()
    K.set_session(sess)

    with open('json_model.json') as f:
        json_str = f.readline().strip()
    model = model_from_json(json_str)

    train_image_tensor, train_labels_tensor = image_processing.batch_inputs(
        data_dir, batch_size, image_size, True, preprocess_operation, num_preprocess_threads,
        num_readers, examples_per_shard, input_queue_memory_factor)

    y = tf.placeholder(tf.int32, shape=(batch_size,))
    x = model.input
    preds = model.output

    train_labels = y
    if labels_offset != 0:
        train_labels = tf.sub(y, labels_offset)

    # check objectives
    if objectives == 'categorical_crossentropy':
        train_labels = tf.one_hot(train_labels, num_classes, axis=-1)
        train_labels = tf.cast(train_labels, tf.float32)
        loss = tf.reduce_mean(categorical_crossentropy(train_labels, preds))
        # this loss is the same as:
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(preds, train_labels))
        acc_value = categorical_accuracy(train_labels, preds)
    elif objectives == 'binary_crossentropy':
        # train_labels: 1-d tensor (batch_size,)
        # preds: 2-d tensor (batch_size, 1)
        train_labels = tf.cast(train_labels, tf.float32)
        preds = tf.reshape(preds, train_labels.get_shape().as_list())
        # or reshape train_labels as:
        # train_labels = tf.reshape(train_labels, [-1,1])
        loss = tf.reduce_mean(binary_crossentropy(train_labels, preds))
        # this loss is the same as:
        # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(preds, train_labels))
        acc_value = binary_accuracy(train_labels, preds)
    else:
        print('No corresponding objectives %s' % objectives)
        exit(-1)

    print(preds.get_shape().as_list())
    print(train_labels.get_shape().as_list())


    sess.run(tf.initialize_all_variables())
    model.load_weights('json_model_weights.h5')

    tf.train.start_queue_runners(sess=sess)
    with sess.as_default():
        for i in range(max_steps):
            tx, ty = sess.run([train_image_tensor, train_labels_tensor])
            cur_loss, cur_accu= sess.run([loss, acc_value],
                                            feed_dict={K.learning_phase(): 0, x:tx, y:ty})
            print cur_loss, cur_accu


if __name__ == '__main__':
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
    model.save('test_call_model_as_layer.h5')
    del model

    print('train')
    worker_train('test_call_model_as_layer.h5', 'data/tfrecords',
                 32, 'crop', [150, 150, 3], 2, 3000,
                 'categorical_crossentropy', 'RMSProp', 0.001, labels_offset=1,
                 num_preprocess_threads=4, num_readers=1, examples_per_shard=100, input_queue_memory_factor=2)
    print('test')
    worker_test('data/tfrecords',
                32, 'crop', [150, 150, 3], 2, 3000,
                'categorical_crossentropy', 'RMSProp', 0.001, labels_offset=1,
                num_preprocess_threads=4, num_readers=1, examples_per_shard=100, input_queue_memory_factor=2)