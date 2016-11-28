import os
import subprocess
import sys
import paramiko
from divide_image_data import *
from build_local_image_data import *
import time
import threading

outlock = threading.Lock()


def ssh_thread(command, server, username, password):
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
    with outlock:
        #output = stdout.read()
        #print output
        for line in iter(lambda:stdout.readline(2048), ""):
            print(line)
    print stderr.read()

    client.close()

def master_data_process(train_directory, validation_directory, output_directory, labels_file,
                        train_shards, validation_shards, num_threads, username, ip_list):
    if os.path.exists(output_directory):
        print('Output directory %s exists' % output_directory)
        return
    num_of_divide = len(ip_list)
    divide_image_data(train_directory, validation_directory, output_directory, labels_file, num_of_divide)

    threads = []

    for i in range(len(ip_list)):
        ip = ip_list[i]
        cur_train_directory = os.path.join(output_directory, 'raw_'+str(i), 'train')
        cur_validation_directory = os.path.join(output_directory, 'raw_' + str(i), 'validation')
        cur_output_directory = os.path.join(output_directory, str(i))
        if not os.path.exists(cur_output_directory):
            os.makedirs(cur_output_directory)

        cur_dir_path = os.path.dirname(os.path.realpath(__file__))
        build_local_image_data_file_path = os.path.join(cur_dir_path, 'build_local_image_data.py')

        command = "python " + build_local_image_data_file_path + " " + cur_train_directory + " " + \
                  cur_validation_directory + " " + cur_output_directory + " " + labels_file + " " + \
                  str(train_shards) + " " + str(validation_shards) + " " + str(num_threads)
        print command
        #ssh
        t = threading.Thread(target=ssh_thread, args=(command, ip_list[i], username, ""))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


        '''
        chan = client.invoke_shell()
        chan.recv(10240)
        chan.send('source ~/.bashrc ; echo $LD_LIBRARY_PATH\n')
        chan.send(command)
        while not chan.exit_status_ready():
            time.sleep(1)
        chan.recv(10240)
        '''

        '''
        stdin, stdout, stderr = client.exec_command("echo $LD_LIBRARY_PATH")
        print stdout.read()
        print stderr.read()
        stdin, stdout, stderr = client.exec_command(command)
        output = stderr.read()
        print output
        # for line in stdout:
        #    print '... ' + line.strip('\n')
        client.close()
        '''


if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        master_data_process(args[1], args[2], args[3], args[4],
                            int(args[5]), int(args[6]), int(args[7]), args[8], args[9].split(','))
    else:
        master_data_process("/home/lpl/Documents/keras-develop/keras-develop/data/train",
                        "/home/lpl/Documents/keras-develop/keras-develop/data/validation",
                        "/home/lpl/Documents/keras-develop/keras-develop/data/tfrecords",
                        "/home/lpl/Documents/keras-develop/keras-develop/data/label.txt",
                        5, 5, 5, "lpl", ["localhost", "localhost"])