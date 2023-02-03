import numpy as np
np.zeros([5, 5, 2])
dict = {}
from dataclasses import dataclass
@dataclass
class annotation:
    label : str
    box:list
lst = []
lst.append(annotation('Car', [0, 497, 286, 652]))
lst.append(annotation('Truck', [4, 504, 282, 668]))

dict['data/images/train/0049/004345.png'] = [lst]
dict

lst

def extract_boxes(lst):
    pass


frame = "'000490': '[364, 239, 513, 367]', '000495': '[315, 253, 504, 397]', '000500': '[320, 254, 540, 416]', '000505': '[423, 269, 716, 482]', '000510': '[633, 270, 1091, 692]', '000515': '[976, 330, 1271, 675]"
004345: [0, 497, 286, 652]
004350: [0, 489, 290, 663]
004355: [4, 504, 282, 668]
004360: [0, 493, 308, 670]
004365: [0, 479, 297, 684]
004370: [0, 482, 304, 669]
type([1,2])
lst
for 
