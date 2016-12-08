import os
import subprocess
import sys

'''
Each worker machine will call generate_tfrecords_file to build their part of tfrecords files
'''
def generate_build_image_data_command(train_directory, validation_directory, output_directory, labels_file,
                                      train_shards, validation_shards, num_threads):
    cur_dir_path = os.path.dirname(os.path.realpath(__file__))
    build_image_data_file_path = os.path.join(cur_dir_path, 'build_image_data.py')

    if train_shards % num_threads != 0 or validation_shards % num_threads != 0:
        print('Error: please make train_shards and validation_shards divisible by num_threads')
        exit(-1)

    command = 'python ' + build_image_data_file_path + ' --train_directory="' + train_directory + \
              '" --validation_directory="' + validation_directory + '" --output_directory="' + output_directory + \
              '" --labels_file="' + labels_file + '" --train_shards=' + str(train_shards) + ' --validation_shards=' + \
              str(validation_shards) + ' --num_threads=' + str(num_threads)
    return command

def generate_tfrecords_file(command, output_directory):
    # require a non-exist or empty output_directory
    if os.path.exists(output_directory):
        if not os.path.isdir(output_directory):
            print('Error: output_directory %s is not a directory' % output_directory)
            exit(-1)
        elif os.listdir(output_directory) != []:
            print('Error: output_directory %s is not empty, content: %s' % (output_directory, os.listdir(output_directory)))
            exit(-1)
    else:
        os.makedirs(output_directory)

    return subprocess.check_call(command, shell=True)

if __name__ == '__main__':
    # arg[1]: train_directory
    # arg[2]: validation_directory
    # arg[3]: output_directory
    # arg[4]: labels_file
    # arg[5]: train_shards
    # arg[6]: validation_shards
    # arg[7]: num_threads

    args = sys.argv
    if len(args) != 8:
        print('Error: check args')
        exit(-1)
    for i in range(1, 5):
        if i == 3:
            continue
        if not os.path.exists(args[i]):
            print('Error: %s does not exist' % args[i])
            exit(-1)
        if i == 4:
            if os.path.isdir(args[i]):
                print('Error: %s is not a file' % args[i])
                exit(-1)
        else:
            if not os.path.isdir(args[i]):
                print('Error: %s is not a directory' % args[i])
                exit(-1)

    command = generate_build_image_data_command(args[1], args[2], args[3], args[4], int(args[5]),
                                                int(args[6]), int(args[7]))
    #return 0 if succeed
    generate_tfrecords_file(command, args[3])

    '''
    command = generate_build_image_data_command('/Users/lipeilun/Documents/Keras-develop/keras-develop/data/train',
                                            '/Users/lipeilun/Documents/Keras-develop/keras-develop/data/validation',
                                            '/Users/lipeilun/Documents/Keras-develop/keras-develop/data/testtf',
                                            '/Users/lipeilun/Documents/Keras-develop/keras-develop/data/label.txt',
                                            5, 5, 5)
    print generate_tfrecords_file(command, '/Users/lipeilun/Documents/Keras-develop/keras-develop/data/testtf/')
    '''