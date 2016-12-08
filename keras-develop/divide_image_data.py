import shutil
import os
import sys
import random
'''
Master will call this function to divide image data for each worker
'''
def divide_image_data(train_directory, validation_directory, output_directory, labels_file, num_of_divide):
    raw_classes = os.listdir(train_directory)
    full_file_dict = {}
    for i in range(len(raw_classes)):
        if raw_classes[i] != '.':
            images_files = os.listdir(os.path.join(train_directory, raw_classes[i]))
            random.shuffle(images_files)
            full_file_dict[raw_classes[i]] = images_files

    divided_file_dict = [{} for i in range(num_of_divide)]
    for classes, files in full_file_dict.items():
        num_files = len(files)
        num_split_files = num_files / num_of_divide
        for i in range(num_of_divide-1):
            divided_file_dict[i][classes] = files[i*num_split_files:(i+1)*num_split_files]
        divided_file_dict[-1][classes] = files[(num_of_divide-1)*num_split_files:]

    for i in range(num_of_divide):
        cur_dir = os.path.join(output_directory, 'raw_'+str(i), 'train')
        os.makedirs(cur_dir)
        for classes, files in divided_file_dict[i].items():
            class_dir = os.path.join(cur_dir, classes)
            os.makedirs(class_dir)
            for j in range(len(files)):
                from_path = os.path.join(train_directory, classes, files[j])
                to_path = os.path.join(class_dir, files[j])
                shutil.copyfile(from_path, to_path)

    # validation set is not splitted
    for i in range(num_of_divide):
        cur_dir = os.path.join(output_directory, 'raw_'+str(i), 'validation')
        if i == 0:
            shutil.copytree(validation_directory, os.path.join(cur_dir))
        else:
            os.makedirs(cur_dir)
            for j in range(len(raw_classes)):
                if raw_classes[j] != '.':
                    os.makedirs(os.path.join(cur_dir, raw_classes[j]))




if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        divide_image_data(args[1], args[2], args[3], args[4], int(args[5]))
    else:
        divide_image_data("/home/lpl/Documents/keras-develop/keras-develop/data/train",
                      "/home/lpl/Documents/keras-develop/keras-develop/data/validation",
                      "/home/lpl/Documents/keras-develop/keras-develop/data/testtf2",
                      "/home/lpl/Documents/keras-develop/keras-develop/data/label.txt",
                      2)