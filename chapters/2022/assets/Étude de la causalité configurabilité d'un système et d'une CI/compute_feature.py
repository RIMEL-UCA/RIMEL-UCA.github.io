def occurences(x):
    return x[1]

def lineNotAComment(line):
    return (not (line[4].startswith('0') or line[4].startswith('(0'))) 

def main():
    with open('cppstats_featurelocations.csv','r') as file :
        list_of_var = {}
        i = 0
        nb_0 = 0
        next(file)
        for line in file:
            var_file = line.strip().split(',')
            if len(var_file)>5:
                if lineNotAComment(var_file):
                    i = i + 1
                    var_list = var_file[5].split(";")
                    for var in var_list:
                        if not var == "defined": 
                            list_of_var.__setitem__(str(var),list_of_var.get(var,0) + 1)
                else :
                    nb_0 = nb_0 + 1
    print("nb 0, " + str(nb_0))
    print("point de variabilite," + str(i))
    for value in sorted(list_of_var.items(), key = lambda x:x[1], reverse =True):
        print(str(value[0]) + "," + str(value[1]))
main()
