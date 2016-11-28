import os
from random import shuffle
import shutil

def subsample(input_path, output_path, num_train_subsample_each_class, num_val_subsample_each_class):
    input_train_path = os.path.join(input_path, 'train')
    # input_val_path = os.path.join(input_path, 'val')
    output_train_path = os.path.join(output_path, 'train')
    output_val_path = os.path.join(output_path, 'val')
    classes = os.listdir(input_train_path)
    for i in range(len(classes)):
        input_train_dir = os.path.join(input_train_path, classes[i])
        # input_val_dir = os.path.join(input_val_path, classes[i])
        output_train_dir = os.path.join(output_train_path, classes[i])
        output_val_dir = os.path.join(output_val_path, classes[i])
        os.makedirs(output_train_dir)
        os.makedirs(output_val_dir)
        train_pics = os.listdir(input_train_dir)
        shuffle(train_pics)
        for j in range(num_train_subsample_each_class):
            shutil.copyfile(os.path.join(input_train_dir, train_pics[j]), os.path.join(output_train_dir, train_pics[j]))
        for j in range(num_train_subsample_each_class, num_train_subsample_each_class + num_val_subsample_each_class):
            shutil.copyfile(os.path.join(input_train_dir, train_pics[j]), os.path.join(output_val_dir, train_pics[j]))




if __name__ == "__main__":
    subsample('/home/lpl/Documents/dataset/ILSVRC2015/Data/CLS-LOC',
              '/home/lpl/Documents/dataset/ILSVRC2015_subsample', 10, 5)