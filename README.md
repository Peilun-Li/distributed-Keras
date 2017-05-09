# Keras Interface for Distributed TensorFlow

This repo uses Keras to add a interface for data parallel in distributed TensorFlow, i.e., cut data into pieces in master node and send them to corresponding worker nodes. Each worker will then read data into TensorFlow's data reading queue and process data into target format, after which the data will be used to train a given model.

This interface assumes that your TensorFlow has a underlying parameter server to share parameters among workers and master node.

## File Structure

docs/ More documents for installation and APIs.

kears/ A Keras version that is used for developing this interface. You can also use the lastest Keras but there maybe inconsistency issues.

keras-develop/ Interface codes, including sample data.

other/ Some testing code for debug, unused for the release version.