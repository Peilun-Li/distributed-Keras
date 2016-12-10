from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import scipy.misc as smp


def batch_inputs(data_dir, batch_size, image_size, train, preprocess_operation, num_preprocess_threads=None,
                 num_readers=1, examples_per_shard=1024, input_queue_memory_factor = 16):
    """Contruct batches of training or evaluation examples from the image dataset.
    Args:
      dataset: instance of Dataset class specifying the dataset.
        See dataset.py for details.
      batch_size: integer
      train: boolean
      num_preprocess_threads: integer, total number of preprocessing threads
      num_readers: integer, number of parallel readers
    Returns:
      images: 4-D float Tensor of a batch of images
      labels: 1-D integer Tensor of [batch_size].
    Raises:
      ValueError: if data is not found
    """
    with tf.name_scope('batch_processing'):
        if train == True:
            subset = 'train'
        else:
            subset = 'validation'

        # Reshape images into these desired dimensions.
        if len(image_size) == 1:
            if isinstance(image_size,list) or isinstance(image_size,tuple):
                image_size = image_size[0]
            height = image_size
            width = image_size
            depth = 3
        elif len(image_size) == 2:
            height = image_size[0]
            width = image_size[1]
            depth = 3
        elif len(image_size) == 3:
            height = image_size[0]
            width = image_size[1]
            depth = image_size[2]
        else:
            print('Wrong image_size dimension %d' % len(image_size))
            exit(-1)

        tf_record_pattern = os.path.join(data_dir, '%s-*' % subset)
        data_files = tf.gfile.Glob(tf_record_pattern)
        if not data_files:
            print('No files found for dataset %s at %s' % (subset, data_dir))
            exit(-1)
        if data_files is None:
            raise ValueError('No data files found for this dataset')

        # Create filename_queue
        if train:
            filename_queue = tf.train.string_input_producer(data_files,
                                                            shuffle=True,
                                                            capacity=16)
        else:
            filename_queue = tf.train.string_input_producer(data_files,
                                                            shuffle=False,
                                                            capacity=1)
        if num_preprocess_threads is None:
            num_preprocess_threads = 4

        if num_readers is None:
            num_readers = 1

        # Approximate number of examples per shard.
        if examples_per_shard is None:
            examples_per_shard = 1024
        if input_queue_memory_factor is None:
            input_queue_memory_factor = 16
        # Size the random shuffle queue to balance between good global
        # mixing (more examples) and memory use (fewer examples).
        # 1 image uses 299*299*3*4 bytes = 1MB
        # The default input_queue_memory_factor is 16 implying a shuffling queue
        # size: examples_per_shard * 16 * 1MB = 17.6GB
        min_queue_examples = examples_per_shard * input_queue_memory_factor
        if train:
            examples_queue = tf.RandomShuffleQueue(
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples,
                dtypes=[tf.string])
        else:
            examples_queue = tf.FIFOQueue(
                capacity=examples_per_shard + 3 * batch_size,
                dtypes=[tf.string])

        # Create multiple readers to populate the queue of examples.
        if num_readers > 1:
            enqueue_ops = []
            for _ in range(num_readers):
                reader = tf.TFRecordReader()
                _, value = reader.read(filename_queue)
                enqueue_ops.append(examples_queue.enqueue([value]))

            tf.train.queue_runner.add_queue_runner(
                tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
            example_serialized = examples_queue.dequeue()
        else:
            reader = tf.TFRecordReader()
            _, example_serialized = reader.read(filename_queue)

        images_and_labels = []
        for thread_id in range(num_preprocess_threads):
            # Parse a serialized Example proto to extract the image and metadata.
            image_buffer, label_index, _ = parse_example_proto(
                example_serialized)
            '''image_preprocessing'''
            image = image_preprocessing(image_buffer, preprocess_operation, train, height, width, depth, thread_id)
            images_and_labels.append([image, label_index, _])

        images, label_index_batch, label_text = tf.train.batch_join(
            images_and_labels,
            batch_size=batch_size,
            capacity=2 * num_preprocess_threads * batch_size)

        images = tf.cast(images, tf.float32)
        images = tf.reshape(images, shape=[batch_size, height, width, depth])

        # Display the training images in the visualizer.
        tf.image_summary('images', images)
        return images, tf.reshape(label_index_batch, [batch_size]), tf.reshape(label_text, [batch_size])


def parse_example_proto(example_serialized):
    """Parses an Example proto containing a training example of an image.
      The output of the build_image_data.py image preprocessing script is a dataset
      containing serialized Example protocol buffers. Each Example proto contains
      the following fields:
        image/height: 462
        image/width: 581
        image/colorspace: 'RGB'
        image/channels: 3
        image/class/label: 615
        image/class/synset: 'n03623198'
        image/class/text: 'knee pad'
        image/object/bbox/xmin: 0.1
        image/object/bbox/xmax: 0.9
        image/object/bbox/ymin: 0.2
        image/object/bbox/ymax: 0.6
        image/object/bbox/label: 615
        image/format: 'JPEG'
        image/filename: 'ILSVRC2012_val_00041207.JPEG'
        image/encoded: <JPEG encoded string>
      Args:
        example_serialized: scalar Tensor tf.string containing a serialized
          Example protocol buffer.
      Returns:
        image_buffer: Tensor tf.string containing the contents of a JPEG file.
        label: Tensor tf.int32 containing the label.
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
          where each coordinate is [0, 1) and the coordinates are arranged as
          [ymin, xmin, ymax, xmax].
        text: Tensor tf.string containing the human-readable label.
      """
    # Dense features in Example proto.
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                                default_value=-1),
        'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
    }
    sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto.
    '''
    feature_map.update(
        {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                     'image/object/bbox/ymin',
                                     'image/object/bbox/xmax',
                                     'image/object/bbox/ymax']})
    '''
    features = tf.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    '''
    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)


    # Note that we impose an ordering of (y, x) just to make life difficult.
    bbox = tf.concat(0, [ymin, xmin, ymax, xmax])

    # Force the variable number of bounding boxes into the shape
    # [1, num_boxes, coords].
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])
    '''

    return features['image/encoded'], label, features['image/class/text']


def image_preprocessing(image_buffer, operation, train, height, width, depth, thread_id=0):
    """Decode and preprocess one image for evaluation or training.
      Args:
        image_buffer: JPEG encoded string Tensor
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
          where each coordinate is [0, 1) and the coordinates are arranged as
          [ymin, xmin, ymax, xmax].
        train: boolean
        thread_id: integer indicating preprocessing thread
      Returns:
        3-D float Tensor containing an appropriately scaled image
      Raises:
        ValueError: if user does not provide bounding box
      """
    image = decode_jpeg(image_buffer)

    '''
    if bbox is None:
        raise ValueError('Please supply a bounding box.')
    '''
    if operation == 'crop':
        image = tf.image.resize_image_with_crop_or_pad(image, height, width)
    elif operation == 'resize':
        image = tf.image.resize_images(image, (height, width))

    '''
    if train:
        image = distort_image(image, height, width, bbox, thread_id)
    else:
        image = eval_image(image, height, width)

    # Finally, rescale to [-1,1] instead of [0, 1)
    image = tf.sub(image, 0.5)
    image = tf.mul(image, 2.0)
    '''
    image = tf.sub(image, 0.5)
    image = tf.mul(image, 2.0)
    return image


def decode_jpeg(image_buffer, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.
    Args:
      image_buffer: scalar string Tensor.
      scope: Optional scope for op_scope.
    Returns:
      3-D float Tensor with values ranging from [0, 1).
    """
    with tf.op_scope([image_buffer], scope, 'decode_jpeg'):
        # Decode the string as an RGB JPEG.
        # Note that the resulting image contains an unknown height and width
        # that is set dynamically by decode_jpeg. In other words, the height
        # and width of image is unknown at compile-time.
        image = tf.image.decode_jpeg(image_buffer, channels=3)

            # print(len(image), len(image[0]), len(image[1]), len(image[2]))
            # print(image)

        # After this point, all image pixels reside in [0,1)
        # until the very end, when they're rescaled to (-1, 1).  The various
        # adjust_* ops all require this range for dtype float.
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        return image


if __name__ == '__main__':
    sess = tf.Session()
    images, labels, texts = batch_inputs('/home/lpl/Documents/dataset/ILSVRC2015_subsample/tfrecords/0', 1, [150, 150, 3], True, 'resize', 1, 1, 100, 2)
    #exit(0)
    print(images)
    print(labels)
    sess.run(tf.initialize_all_variables())
    tf.train.start_queue_runners(sess=sess)
    cnt = 1
    #while True:
    for i in range(1):
        print(cnt)
        cnt += 1
        image, label, text = sess.run((images,labels, texts))
        print(label, text)
        print(image)
        img = smp.toimage(image[0])
        img.show()
        '''
        for i in range(10):
            print(cnt)
            cnt += 1
            print(image)
            ima = sess.run((image))
            print(ima)

            img = smp.toimage(ima)
            img.show()
        '''
    #print(len(image), len(image[0]), len(image[1]), len(image[2]))
    #print(image)