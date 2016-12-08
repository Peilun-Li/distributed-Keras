from master_data_process import *
import paramiko
import threading
import ast
import sys
'''
Master entry function
'''
outlock_master = threading.Lock()

def line_buffered(f):
    line_buf = ""
    while not f.channel.exit_status_ready():
        line_buf += f.read(1)
        if line_buf.endswith('\n'):
            yield line_buf
            line_buf = ''

def ssh_thread_start(command, server, username, password, output=False):
    pre_command = """
            . ~/.profile ;
            . ~/.bashrc ;
            """

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if password == None or len(password) == 0:
        client.connect(server, username=username)
    else:
        client.connect(server, username=username, password=password)


    stdin, stdout, stderr = client.exec_command(pre_command + command)
    """
    with outlock:
        #output = stdout.read()
        #print output
        for line in iter(lambda:stdout.readline(), ""):
            #if output:
            if True:
                print line
            else:
                pass
    """
    for l in line_buffered(stdout):
        outlock_master.acquire()
        print l
        sys.stdout.flush()
        outlock_master.release()
    #if output:
    print stderr.read()

    client.close()



def master_start(model_path, train_directory, validation_directory, output_directory, labels_file,
                 train_shards, validation_shards, num_divide_threads, username, ip_list, batch_size, preprocess_operation,
                 image_size, num_classes, train_steps, val_stpes, val_interval,
                 objectives, labels_offset = 0, num_preprocess_threads=None, num_readers=1,
                 examples_per_shard=1024, input_queue_memory_factor = 16):
    master_data_process(train_directory, validation_directory, output_directory, labels_file,
                        train_shards, validation_shards, num_divide_threads, username, ip_list)

    cur_dir_path = os.path.dirname(os.path.realpath(__file__))
    worker_train_path = os.path.join(cur_dir_path, 'worker_train_keras.py')

    threads = []

    for i in range(len(ip_list)):
        if i == 0:
            command = ['python', worker_train_path, model_path, os.path.join(output_directory, str(i)), str(batch_size),
                   preprocess_operation, str(image_size).replace(" ", ""), str(num_classes), str(train_steps), str(val_stpes),
                   str(val_interval), objectives, 'True', str(labels_offset), str(num_preprocess_threads),
                   str(num_readers), str(examples_per_shard), str(input_queue_memory_factor)]
        else:
            command = ['python', worker_train_path, model_path, os.path.join(output_directory, str(i)), str(batch_size),
                       preprocess_operation, str(image_size).replace(" ", ""), str(num_classes), str(train_steps), str(val_stpes),
                       str(val_interval), objectives, 'False', str(labels_offset), str(num_preprocess_threads),
                       str(num_readers), str(examples_per_shard), str(input_queue_memory_factor)]
        command = ' '.join(command)
        print command

        if i == 0:
            t = threading.Thread(target=ssh_thread_start, args=(command, ip_list[i], username, "", True))
        else:
            t = threading.Thread(target=ssh_thread_start, args=(command, ip_list[i], username, "", False))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

if __name__ == '__main__':
    args = sys.argv
    if len(args) > 1:
        master_start(args[1],
                     args[2],
                     args[3],
                     args[4],
                     args[5],
                     int(args[6]), int(args[7]), int(args[8]), args[9], args[10].split(","),
                     int(args[11]), args[12], ast.literal_eval(args[13]), int(args[14]), int(args[15]), int(args[16]), int(args[17]),
                     args[18], int(args[19]),
                     int(args[20]), int(args[21]), int(args[22]), int(args[23]))
    else:
        '''
        master_start('/home/lpl/Documents/keras-develop/keras-develop/resnet50.h5',
                     "/home/lpl/Documents/dataset/ILSVRC2015_subsample/train",
                     "/home/lpl/Documents/dataset/ILSVRC2015_subsample/val",
                     "/home/lpl/Documents/dataset/ILSVRC2015_subsample/tfrecords",
                     "/home/lpl/Documents/dataset/ILSVRC2015_subsample/imagenet_label.txt",
                     50, 50, 5, "lpl", ["localhost"],
                     16, 'resize', [224, 224, 3], 1000, 1000, 50, 100,
                     'categorical_crossentropy', labels_offset=1,
                     num_preprocess_threads=4, num_readers=1, examples_per_shard=200, input_queue_memory_factor=2)
        exit(0)
        '''
        '''
        master_start('/home/lpl/Documents/keras-develop/keras-develop/inception_v3.h5',
                     "/home/lpl/Documents/dataset/ILSVRC2015_subsample/train",
                     "/home/lpl/Documents/dataset/ILSVRC2015_subsample/val",
                     "/home/lpl/Documents/dataset/ILSVRC2015_subsample/tfrecords",
                     "/home/lpl/Documents/dataset/ILSVRC2015_subsample/imagenet_label.txt",
                     50, 50, 5, "lpl", ["localhost"],
                     16, 'resize', [299, 299, 3], 1000, 1000, 50, 100,
                     'categorical_crossentropy', labels_offset=1,
                     num_preprocess_threads=4, num_readers=1, examples_per_shard=200, input_queue_memory_factor=2)
        exit(0)
        '''

        master_start('/home/lpl/Documents/keras-develop/keras-develop/predefined_model.h5',
                 "/home/lpl/Documents/keras-develop/keras-develop/data/train",
                        "/home/lpl/Documents/keras-develop/keras-develop/data/validation",
                        "/home/lpl/Documents/keras-develop/keras-develop/data/tfrecords",
                        "/home/lpl/Documents/keras-develop/keras-develop/data/label.txt",
                        5, 5, 5, "lpl", ["localhost", "localhost"],
                        100, 'resize', [150,150,3], 2, 300, 10, 20,
                        'categorical_crossentropy', labels_offset=1,
                        num_preprocess_threads=4, num_readers=1, examples_per_shard=100, input_queue_memory_factor=2)

