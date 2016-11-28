import paramiko


if __name__ == "__main__":
    client = paramiko.SSHClient()
    server = "localhost"
    username = "hadoop"
    password = "hadoop"
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, username=username, password=password)
    stdin, stdout, stderr = client.exec_command('cat examples.desktop')
    output = stdout.read()
    print output
    #for line in stdout:
    #    print '... ' + line.strip('\n')
    client.close()
