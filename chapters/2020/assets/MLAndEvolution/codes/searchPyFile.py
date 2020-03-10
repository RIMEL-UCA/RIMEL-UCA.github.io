res = []

def searchPyFile(file):
    with open(file) as fp:
       line = fp.readline()
       while line:
           #print(line)
           res.append(line.split(" "))
           line = fp.readline()
    print(res)


searchPyFile("test.txt")
