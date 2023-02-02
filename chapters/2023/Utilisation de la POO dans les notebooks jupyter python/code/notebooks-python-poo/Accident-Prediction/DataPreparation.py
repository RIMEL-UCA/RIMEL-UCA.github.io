import numpy as np
import pandas as pd
import opencv as cv
dec = {}
objects = ['Car', 'Truck', 'Pedestrian']
for line in open('./annotations/0041.txt'):
    if line.split(',')[0] in objects:
        obj = line.split(',')[0]
        frames = line[line.find('{') + 1: line.find('}')]
        for f in frames.split("',"):
            f = f.strip().replace("'", "")
            img = f.split(':')[0]
            coor = f.split(':')[1]
#             print(obj, img, coor)
            dec[obj+img] = coor
dec
frame = "'000490': '[364, 239, 513, 367]', '000495': '[315, 253, 504, 397]', '000500': '[320, 254, 540, 416]', '000505': '[423, 269, 716, 482]', '000510': '[633, 270, 1091, 692]', '000515': '[976, 330, 1271, 675]"

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

