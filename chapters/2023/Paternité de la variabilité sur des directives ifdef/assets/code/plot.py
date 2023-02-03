import json
import ipywidgets as widgets
import plotly
import pandas as pd
from anytree import PostOrderIter
from anytree.importer import DictImporter

f = open("out.json", "r")
loaded_json_dic = json.load(f)


new_dict = {"name":"City", "children":[]}

def find_good_children_array(splitted_key):
    arr = []


    for key in splitted_key:
        new_arr = []
        child = {"name": key, "children": new_arr}
        arr.append(child)
        arr = new_arr


    return arr


for key in loaded_json_dic.keys():

    #splitted_key = key.split("/")

    parent = {}
    parent["name"] = key
    parent["children"] = []

    #arr = find_good_children_array(splitted_key)

    for author_commits in loaded_json_dic[key]:
        child = {"name": author_commits[0], "value": author_commits[1]}
        parent["children"].append(child)
    
    new_dict["children"].append(parent)



#imports dictonary in a tree form
importer = DictImporter()
root = importer.import_(new_dict)


size = []
name = []
parent = []
level = []

def format(node):
  for i in node.children:
    #check if node as attribute value
    if hasattr(i, "value") == False:
      format(i)
    v = i.value
    #check if node parent as attribute value
    if hasattr(i.parent, "value"):
      i.parent.value += v
    #if node parent doesn't have a value set to same val as child
    elif hasattr(i.parent, "value")== False:
      i.parent.value = v

    level.append(len(i.ancestors))
    name.append(i.name)
    parent.append(i.parent.name)
    size.append(i.value)
    
format(root)

#append attributes for root
level.append(0)
name.append(root.name)
parent.append("")
size.append(root.value)

#create df
df = pd.DataFrame()
df['parent'] = parent
df['name'] = name
df['value']= size
df['level'] = level

#slider funtion
def update(sliderVal):
    fig = plotly.graph_objs.Figure()
    fig.add_trace(plotly.graph_objs.Treemap(
            labels = df[df['level']<sliderVal]['name'],
            values = df[df['level']<sliderVal]['value'],
            parents = df[df['level']<sliderVal]['parent'] 
        ))
    fig.update_traces(root_color="#f2f1f1")
    fig.update_layout(width = 900, height = 900)
    fig.show()

        #create slider widget

widgets.interact(update, sliderVal = (10, 10))