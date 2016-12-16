# Keras Development Manual

## API of master_start.py

master_start(predefined_keras_model_path, train_directory, validation_directory, output_directory, labels_file, train_shards, validation_shards, num_divide_threads, username, ip_list, batch_size, preprocess_operation, image_size, num_classes, train_steps, val_stpes, val_interval, objectives, labels_offset = 0, num_preprocess_threads=None, num_readers=1, examples_per_shard=1024, input_queue_memory_factor = 16)

- predefined_keras_model_path: user defined keras model

- train_directory: directory that contains train images

- validation_directory: directory that contains validation images

- output_directory: output directory of pre-processed tfrecords files, note that data will be pre-processed (from raw images to tfrecords) only if the directory does not exist. If the directory exists, the program will assume that the tfrecords files already exist and skip the data pre-process (from raw images to tfrecords) process.

- labels_file: a txt file contains all label classes

- train_shards: the number of training tfrecords shards for each worker

- validation_shards: the number of validation tfrecords shards for each worker

- num_divide_threads: the number of threads to build tfrecords files for each worker

- username: ssh username

- ip_list: workers' ip list, the first ip is the master node

- batch_size: training and validation batch size

- preprocess_operation: currently support 'resize' and 'crop'. Note that in this demo only 'resize' works (because the demo uses v0.10 tensorflow, and those operations are developed in v0.11 tensorflow, and for the 'crop' method, there's some changes on tensorflow api between v0.10 and v0.11). If error occur when using 'resize', please check line 226 of image_processing.py. In v0.10 it should be image = tf.image.resize_images(image, height, width), and in v0.11 it should be image = tf.image.resize_images(image, (height, width))

- image_size: [height, width, channels]

- num_classes: the number of class labels

- train_steps: the number of training steps, each step will run a image batch (of batch_size)

- val_steps: the number of validation steps, each step will run a image batch (of batch_size)

- val_interval: the number of training steps between each validation process. For each validation process, the program will run val_steps steps on validation data and calculate the mean of outputs as validation results.

- objectives: currently support 'categorical_crossentropy' and 'binary_crossentropy'

- labels_offset: the background class is set as label 0 by default, so that the acctual labels start from 1. If you don't want background class, set labels_offset to 1, otherwise 0

- num_preprocess_threads: number of data preprocess (such as crop and resize) threads

- num_readers: number of parallel readers to read tfrecords

- examples_per_shard: approximate number of examples per tfrecord shard.

- input_queue_memory_factor: the size of tensorflow's tfrecords shuffling queue will be examples_per_shard * input_queue_memory_factor * per_image_disk_size

## Output

Logs of loss and accuray will be outputted during training and validation. The final trained model will be save as trained_model.h5 currently.

## File Description

- build_image_data.py: mainly transferred from tensorflow inception example. Convert raw image data to tfrecords files.

- build_local_image_data.py: each node will run this script, whilch will call build_image_data.py internally to build tfrecords files from raw images.

- define_model.py: a demo of user-defined model

- define_inception_v3_model.py： a demo of inception v3 model

- define_resnet50_model.py: a demo of resnet50 model

- divide_image_data.py: master node will run this script to allocate raw images files for each node.

- image_processing.py: mainly transferred from tensorflow inception example. Load tfrecords files into tensorflow queues and do image preprocessing (such as resize and crop)

- master_data_process.py： encapsulation of divide_image_data.py and build_local_image_data.py. Master node will call this script to convert raw images data to tfrecords file for each worker.

- master_start.py: program entrance

- worker_train_keras.py: train models and do evaluation.

- worker_train.py: an alternative way to train models and do evaluation. No use now.

