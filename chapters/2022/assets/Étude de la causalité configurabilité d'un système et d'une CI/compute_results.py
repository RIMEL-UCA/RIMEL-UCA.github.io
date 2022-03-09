import matplotlib.pyplot as plt

def set_of_feature(dictionary):
    global NB_VAR
    fileobj = open("cppstats_featurelocations.csv",'r')
    list_of_var = {}
    for line in fileobj:
        line_str = line.strip().split(",")[-1]  
        const_str_list = line_str.split(";")
        for i in range(len(const_str_list)):
            if const_str_list[i] != '':
                if const_str_list[i] in list_of_var :
                    list_of_var[const_str_list[i]] = list_of_var[const_str_list[i]] +1
                else:
                    list_of_var[const_str_list[i]] = 1
                type_of_var = dict_type_var(dictionary, str(const_str_list[i]))
                
    fileobj.close()
    return list_of_var

def dict_type_var(dictionary, var):
    for key in dictionary.keys() :
        for value  in dictionary[key]:
            if value in var.lower():
                #print(key," ",var)
                return key
    return "autre"

def dictionary_map():
    fileobj = open('dictionary.csv','r')
    dictionnary = {}
    for line in fileobj:
        line_str = line.strip().split(",")
        type_of_var = line_str[0]
        dictionnary[type_of_var] = []
        for variable in line_str[1:]:
            dictionnary[type_of_var].append(str(variable).lower())
    return dictionnary
    fileobj.close()

def number_of_constant(dictionary, features):
    ARCH_TYPE_COUNT = {'autre': 0}
    keys = dictionary.keys()
    for val in keys: 
        ARCH_TYPE_COUNT[val] = 0
    for val in features.keys():
        type_of_var = dict_type_var(dictionary, str(val))
        ARCH_TYPE_COUNT[type_of_var] = ARCH_TYPE_COUNT[type_of_var] + 1
    return ARCH_TYPE_COUNT

def number_of_variation(dictionary, features):
    ARCH_TYPE_COUNT = {'autre': 0}
    keys = dictionary.keys()
    for value in keys: 
        ARCH_TYPE_COUNT[value] = 0
    #print(features)
    for val in features:
        type_of_var = dict_type_var(dictionary, str(val))
        ARCH_TYPE_COUNT[type_of_var] = ARCH_TYPE_COUNT[type_of_var] + features[val]
    return ARCH_TYPE_COUNT
def plot_donut_chart(results) :
    labels = []
    sizes = []
    colors = ["#1A78C9","#416F96","#08FCED","#FD6947","#C9221A","#C97324","#967D66","#86FDB0","#24C930", "#85FFF7"]
    for val in results : 
        if results[val] != 0 :
            labels.append(val)
            sizes.append(results[val])
    plt.pie(sizes,  labels=labels, colors=colors, autopct='%1.1f%%')

    #draw a circle at the center of pie to make it look like a donut
    centre_circle = plt.Circle((0,0),0.75,color='white', fc='white',linewidth=1.25)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)


    # Set aspect ratio to be equal so that pie is drawn as a circle.
    plt.axis('equal')
    plt.show()  

def main():
    dictionary = dictionary_map()
    features = set_of_feature(dictionary)
    variaitions = number_of_variation(dictionary,features)

    constant =number_of_constant(dictionary,features)

    plot_donut_chart(variaitions)
    plot_donut_chart(constant)
    #print("number of variation :",NB_VAR)
    #print("number of constant :",len(features))
    #print("number of architecture :",NB_ARCH)
    #print("number of os :", NB_OS)
main()
