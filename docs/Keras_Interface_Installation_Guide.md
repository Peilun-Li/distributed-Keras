# Keras Interface Installation Guide

## Obtaining Keras Interface

Keras Interface is a group of Python scripts. You can get it from [here](https://github.com/Hwhitetooth/Poseidon-TensorFlow/tree/d0.10-keras/keras-develop).

If you are using Orca, you can find the Keras Interface from
```
/users/hzhang2/projects/tensorflow/keras-develop
```

## Installing Poseidon-TensorFlow

You can install Poseidon-TensorFlow following the steps [here](https://github.com/petuum/poseidon-doc/blob/master/docs/installation.md). If you are using Orca, you can simply run the following commands on EACH worker and master node.

```
cd /users/hzhang2/projects/tensorflow/tf-keras/_python_build/
sudo python setup.py develop
```

## Installing Dependencies

Keras Interface depends on:

  - [Keras](https://keras.io/#installation)
  - [cryptography](https://cryptography.io/en/latest/installation/)
  - [Paramiko](http://www.paramiko.org/installing.html)

You can find the installation steps of dependencies from there corresponding installation page. If you are using Orca, you can run the following commands for EACH worker and master node. 

```
sudo pip install scipy
sudo pip install pyyaml --no-use-wheel
sudo apt-get install -y libhdf5-serial-dev
sudo pip install h5py
sudo pip install keras --no-use-wheel
sudo apt-get install -y build-essential libssl-dev libffi-dev python-dev
sudo apt-get install -y python-cffi
sudo pip install cryptography --no-use-wheel
sudo pip install paramiko --no-use-wheel
```

## Running a Demo

Now you can try a demo using Keras Interface. First, let's start Poseison-Tensorflow:

```
cd /users/hzhang2/projects/tensorflow/keras-develop
bash run-poseidon-keras.sh 2 start keras
```

Then let's run the demo using the following command. Note that you should change orca35.orca.pdl.cmu.edu, orca15.orca.pdl.cmu.edu in the command to corresponding node address, and the first address needs to be the master node. You can find more details of the API of master_start.py in Keras Development Manual.


```
python /users/hzhang2/projects/tensorflow/keras-develop/master_start.py /users/hzhang2/projects/tensorflow/keras-develop/predefined_model.h5 /users/hzhang2/projects/tensorflow/keras-develop/data/train /users/hzhang2/projects/tensorflow/keras-develop/data/validation /users/hzhang2/projects/tensorflow/keras-develop/data/tfrecords /users/hzhang2/projects/tensorflow/keras-develop/data/label.txt 5 5 5 hzhang2 orca35.orca.pdl.cmu.edu,orca15.orca.pdl.cmu.edu 100 resize [150,150,3] 2 300 10 20 categorical_crossentropy 1 4 1 100 2
```

When runing the demo, you can check the status using:

```
bash run-poseidon-keras.sh 2 check
```

After finished the demo, you can stop Poseidon-TensorFlow using:

```
bash run-poseidon-keras.sh 2 stop
```

## More Demos

You can define much more complex neural networks using Keras. There are two scripts "define_inception_v3_model.py" and "define_resnet50_model.py" along with Keras interface's codes. Simply run them you can get predefined inception v3 and resnet50 model. Then you can run something like:

```
python /users/hzhang2/projects/tensorflow/keras-develop/master_start.py /users/hzhang2/projects/tensorflow/keras-develop/inception_v3.h5 /users/hzhang2/projects/tensorflow/keras-develop/data/ILSVRC2015_subsample/train /users/hzhang2/projects/tensorflow/keras-develop/data/ILSVRC2015_subsample/val /users/hzhang2/projects/tensorflow/keras-develop/data/ILSVRC2015_subsample/tfrecords /users/hzhang2/projects/tensorflow/keras-develop/data/ILSVRC2015_subsample/imagenet_label.txt 50 50 5 hzhang2 orca03.orca.pdl.cmu.edu,orca19.orca.pdl.cmu.edu 32 resize [299,299,3] 1000 1000 50 100 categorical_crossentropy 1 4 1 100 2
```

or

```
python /users/hzhang2/projects/tensorflow/keras-develop/master_start.py /users/hzhang2/projects/tensorflow/keras-develop/resnet50.h5 /users/hzhang2/projects/tensorflow/keras-develop/data/ILSVRC2015_subsample/train /users/hzhang2/projects/tensorflow/keras-develop/data/ILSVRC2015_subsample/val /users/hzhang2/projects/tensorflow/keras-develop/data/ILSVRC2015_subsample/tfrecords /users/hzhang2/projects/tensorflow/keras-develop/data/ILSVRC2015_subsample/imagenet_label.txt 50 50 5 hzhang2 orca03.orca.pdl.cmu.edu,orca19.orca.pdl.cmu.edu 32 resize [224,224,3] 1000 1000 50 100 categorical_crossentropy 1 4 1 100 2
```
