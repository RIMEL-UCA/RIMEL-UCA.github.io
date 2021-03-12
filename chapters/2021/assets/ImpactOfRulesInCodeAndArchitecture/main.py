import os
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        print("Specify the directory.\n\tmain.py <directory>")
        sys.exit(1)

    if not os.path.exists("rules/"):
        os.makedirs("rules/")


    with open("input_files/input.txt") as file:
        rules = {}
        for line in file:
            array = line.rstrip().split(',')
            rules[array[0]] = array[1:]

        report_file = open("rules/report.csv", "w")
        report_file.write("Rule,Keyword,Occurence\n")

        for rule, keywords in rules.items():
            if not os.path.exists("rules/" + rule + "/lines/"):
                os.makedirs("rules/" + rule + "/lines/")


            for keyword in keywords:
                sum = 0
                occurences = {}
                command = "find " + sys.argv[1] + " -type f ! -path './rules/*' -print | xargs grep -in '"
                command += keyword.replace(" ", "\\ ")
                command += "' | grep -v ':0$'"
                process = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
                result = process.stdout.decode("utf-8")

                with open("rules/" + rule + "/lines/" + keyword + ".txt", "w") as f:
                    f.write(result)

                for line in result.split("\n"):
                    if line == "":
                        continue

                    splitted = line.split(":")
                    if splitted[0] not in occurences:
                        occurences[splitted[0]] = 0

                    occurences[splitted[0]] += 1

                with open("rules/" + rule + "/" + keyword + ".txt", "w") as f:
                    for k, v in occurences.items():
                        sum += v
                        f.write(k + "," + str(v) + "\n")

                    f.write(str(sum))
                    report_file.write(rule +","+keyword+","+str(sum)+"\n")



if __name__ == '__main__':
    main()
