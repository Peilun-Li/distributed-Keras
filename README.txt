# Keras-develop
Usage:

0. Swap in the experiment poseidon-tf-keras, wait until the startup script finishes, approximately 10 minutes. [it seems that the startup script can't pip install, so we need to install some packages mannually for each node, see step 1.]

1. For each node, login and run following [ssh hzhang2@orca12.orca.pdl.cmu.edu, change 12 to node num]:

cd ~/projects/tensorflow/tf-keras/_python_build/
sudo python setup.py develop

sudo pip install scipy
sudo pip install pyyaml --no-use-wheel
sudo apt-get install -y libhdf5-serial-dev
sudo pip install h5py
sudo pip install keras --no-use-wheel
sudo apt-get install -y build-essential libssl-dev libffi-dev python-dev
sudo apt-get install -y python-cffi
sudo pip install cryptography --no-use-wheel
sudo pip install paramiko --no-use-wheel

cd /users/hzhang2/projects/tensorflow/keras-develop/keras
sudo python setup.py install

2. Login to gatekeeper or the master node [master node is the first ip/address in the following command, i.e., h0 in orca]:

cd ~/projects/tensorflow/keras-develop
bash run-poseidon-keras.sh 2 start keras


3. For master node, run the demo [change orca35.orca.pdl.cmu.edu,orca15.orca.pdl.cmu.edu in the command to corresponding node address, note that the first address needs to be the master node, i.e., h0 in orca]:

python /users/hzhang2/projects/tensorflow/keras-develop/master_start.py /users/hzhang2/projects/tensorflow/keras-develop/predefined_model.h5 /users/hzhang2/projects/tensorflow/keras-develop/data/train /users/hzhang2/projects/tensorflow/keras-develop/data/validation /users/hzhang2/projects/tensorflow/keras-develop/data/tfrecords /users/hzhang2/projects/tensorflow/keras-develop/data/label.txt 5 5 5 hzhang2 orca35.orca.pdl.cmu.edu,orca15.orca.pdl.cmu.edu 100 resize [150,150,3] 2 300 10 20 categorical_crossentropy 1 4 1 100 2


Notes:

1. API of master_start.py:

master_start(predefined_keras_model_path, train_directory, validation_directory, output_directory, labels_file, train_shards, validation_shards, num_divide_threads, username, ip_list, batch_size, preprocess_operation, image_size, num_classes, train_steps, val_stpes, val_interval, objectives, labels_offset = 0, num_preprocess_threads=None, num_readers=1, examples_per_shard=1024, input_queue_memory_factor = 16)

predefined_keras_model_path: user defined keras model, defined in "define_model.py" in this demo and saved as "predefined_model.h5"

train_directory: directory that contains train images

validation_directory: directory that contains validation images

output_directory: directory of pre-processed tfrecords files, note that data will be pre-processed (from raw images to tfrecords) only if the directory does not exist. If the directory exists, the program will assume that the tfrecords files already exist and skip the data pre-process (from raw images to tfrecords) process.

labels_file: a txt file contains all label class

train_shards: the number of training tfrecords shards for each worker

validation_shards: the number of validation tfrecords shards for each worker

num_divide_threads: the number of threads to build tfrecords files for each worker

username: ssh username

ip_list: workers' ip list

batch_size: training and validation batch size

preprocess_operation: currently support 'resize' and 'crop'. Note that in this demo only 'resize' works (because the demo uses v0.10 tensorflow, and those operations are developed in v0.11 tensorflow, and for the 'crop' method, there's some changes on tensorflow api between v0.10 and v0.11)

image_size: 3-D nums

num_classes: the number of class labels

train_steps: the number of training steps, each step will run a image batch (of batch_size)

val_steps: the number of validation steps, each step will run a image batch (of batch_size)

val_interval: the number of steps between each validation process. For each validation process, the program will run val_steps steps on validation data and calculate the mean of outputs as validation results.


objectives: currently support 'categorical_crossentropy' and 'binary_crossentropy'

labels_offset: the background class is set as label 0 by default, if you don't want background class, set labels_offset to 1

num_preprocess_threads: number of data preprocess (such as crop and resize) threads

num_readers: number of parallel readers to read tfrecords

examples_per_shard: approximate number of examples per tfrecord shard.

input_queue_memory_factor: the size of tensorflow's tfrecords shuffling queue will be examples_per_shard * input_queue_memory_factor * per_image_disk_size


2. output of the demo:

The demo will output something like:

[0.38859862, 0.81999993]

[0.49113157, 0.79999995]

[0.40444535, 0.83999997]

('val: ', [0.516871988773346, 0.7479999661445618])

Each tuple is an output of a train of validation batch, and the output format is [loss, accuracy]. Note that there's some delay to get output from ssh, so after starting running the demo, the output will be empty for some time and then suddenly outputs for many batches will be displayed.

Currently we only show the output of master node.

3. 

If runnning "bash run-poseidon-keras.sh 2 check" (in directory /users/hzhang2/projects/tensorflow/keras-develop/) when the demo is running, you can find there's a python process for each node running the demo. And if login to each node and run "nvidia-smi", you can find there's a python process using about 4GB GPU Memory for each node.

4. Dataset:

The dataset of the demo is chosen from kaggle's dog and cat dataset. There are 3000 train images and 1000 validation images for each class.

5. Output:

The trained model will be save in "/users/hzhang2/projects/tensorflow/keras-develop/trained_model.h5"


Files description:

build_image_data.py: Mainly transfer from tensorflow inception example. Convert raw image data to tfrecords files.

build_local_image_data.py: Each node will run this script, whilch will call build_image_data.py internally to build tfrecords files from raw images.

define_model.py: user-defined model file

divide_image_data.py: master node will run this script to allocate raw images files for each node.

image_processing.py: Mainly transfer from tensorflow inception example. Load tfrecords files into tensorflow queues and do image preprocessing (such as resize and crop)

master_data_process.py： encapsulation of divide_image_data.py and build_local_image_data.py

master_start.py: program entrance

worker_train_keras.py: Train models and do evaluation.

worker_train.py: Alternative way to train models and do evaluation. No use in this demo.
