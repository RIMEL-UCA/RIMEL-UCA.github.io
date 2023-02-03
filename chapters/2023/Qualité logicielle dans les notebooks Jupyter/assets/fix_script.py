import sys

def main() :
    print(sys.argv)
    if(len(sys.argv)<2) :
        return
    file_name = sys.argv[1]
    print(file_name)

    file1 = open(file_name, "r")
    lines = file1.readlines()
    file1.close()

    file2 = open(file_name, "w")
    for line in lines :
        if "%matplotlib inline" in line :
            line = ""
        file2.write(line)

    file2.close()


main()