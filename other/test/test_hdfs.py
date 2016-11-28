import pydoop
import pydoop.hdfs as hdfs

if __name__ == "__main__":
    print hdfs.ls("/user")
    print hdfs.ls(".")
    files = hdfs.ls(".")
    text = hdfs.load(files[0])
    print text[0:20]
    print hdfs.path.isdir("/user")
    basename = hdfs.path.basename(files[0])

    #hdfs.get(basename,"/tmp/"+basename)
    #with open("/tmp/"+basename) as f:
    #    print f.read()

    #hdfs.put("/tmp/"+basename, basename+".copy")
    #print hdfs.load(basename+".copy")

    print hdfs.ls(".")