# Keras Interface Installation Guide for AWS

## Notes

This guide assumes that you have already have Poseidon-TensorFlow installed on an AWS instance.

## SSH to AWS instance

Use [ssh-agent](https://docs.google.com/document/d/19SB3K7bWrhHJ-wNfyzG9Xzous2NApEhxBZ9cVWGYURg/edit?usp=sharing) to connect to AWS instance. Don't forget to add -A flag when ssh’ing into the machine. Specifically, run the commands to ssh to AWS machine.

```
eval `ssh-agent -s`
ssh-add /path/to/your_aws_private_key.pem
ssh -A ubuntu@aws_instance_ip 
```

## Obtaining Keras Interface

Keras Interface is a group of Python scripts. You can get the Keras Interface from [here](https://bitbucket.org/petuuminc/poseidon/src/dbe5ea11f6b9?at=d0.10-keras), which is a directory called "keras-develop".

If there's a /home/ubuntu/keras-develop directory on the AWS instance, you can use it directly.

## Installing Dependencies

Keras Interface depends on:

  - [Keras](https://keras.io/#installation)
  - [cryptography](https://cryptography.io/en/latest/installation/)
  - [Paramiko](http://www.paramiko.org/installing.html)

You can find the installation steps of dependencies from there corresponding installation page. Specifically, you can run the following commands for EACH worker and master node. 

```
sudo python -m pip install --upgrade pip
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

## Modifying ~/.profile

Copy contents from ~/.bashrc to ~/.profile, in other words, add the following commands to ~/.profile. This is because Python's Paramiko ssh client can only get information from ~/.profile.

```
export CUDA_HOME="/usr/local/cuda"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64"
```

## Running a Demo

We can try a demo using Keras Interface. This a demo for 2-class classification, and the dataset is subsampled from [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data).

First, start master and workers following Poseidon-TensorFlow's instructions. Actually Keras Interface can run in any kind of TensorFlow -- even if the master and workers haven't been started.

Then let's generate user defined Keras model. In this demo, we can run the following command to get a "predefined_model.h5" file.

```
python define_model.py
```

Finally we can run the demo using the following command. Note that we should change 172.31.28.241 in the command to corresponding node address. You can find more details of the API of master_start.py in [Keras Development Manual](https://github.com/petuum/poseidon-doc/blob/master/docs/Keras_Development_Manual.md).

```
python /home/ubuntu/keras-develop/master_start.py /home/ubuntu/keras-develop/predefined_model.h5 /home/ubuntu/keras-develop/data/train /home/ubuntu/keras-develop/data/validation /home/ubuntu/keras-develop/data/tfrecords /home/ubuntu/keras-develop/data/label.txt 5 5 5 ubuntu 172.31.28.241 100 resize [150,150,3] 2 300 10 20 categorical_crossentropy 1 4 1 100 2
```

The demo will output something like the following format:

```
ip-address time [loss, accuracy]
```

If you want to try the demo in a distributed environment, simply change 172.31.28.241 to something like 172.31.28.241,172.31.28.242. Note that the first address needs to be the master node, and the interface assumes you have a NFS like storage system to share files.

## More Demos

You can define much more complex neural networks using Keras. There are two more examples "define_inception_v3_model.py" and "define_resnet50_model.py" in Keras Interface. Simply run them you can get predefined inception v3 and resnet50 model. You can find more details from the two examples and Keras documents (such as [this](https://keras.io/optimizers/)) on how to define complex models and set hyper parameters.