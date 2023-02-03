#!/usr/bin/env python
# coding: utf-8

# In[19]:


from adjustText import *
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sb
import numpy as np
import pandas as pd
from math import pi

sb.set(font_scale=0.7)
pd.set_option('display.width', 800)
pd.set_option('display.max_columns', 200)
plt.style.use('ggplot')
rcParams['figure.figsize'] = (6,6)
rcParams['figure.dpi'] = 300
rcParams['font.size'] = 8
rcParams['font.family'] = 'DejaVu Sans'

def average_polar_vec(angles,values,scale=3):
    '''average_polar_vec()
        Adds up all of the polar vectors given by angles and magnitudes
        and returns the sum angle and magnitude. 
        
        angles: angles of vectors from 0,2pi
        values: magnitude (r) of vectors 
        scale: scaling factor (for output), a divisor on the output magnitude. Default = 3.
    '''
    #step 1: get all of the vector x and y magnitudes
    vectors_x = [values[ix]*np.cos(angles[ix]) for ix in range(0,len(values[:-1]))]
    vectors_y = [values[ix]*np.sin(angles[ix]) for ix in range(0,len(values[:-1]))]
    f_x = sum(vectors_x)
    f_y = sum(vectors_y)
    vector_r = np.sqrt(f_x**2+f_y**2)/scale
    vector_ang = np.arctan(f_y/f_x)
    if f_x<0:
        vector_ang += np.pi
    return vector_ang, vector_r

results = {'gene1': {'ast': (0.15285807433166704, 0.024996159136543922),
              'end': (0.20495061460069067, 0.002529681321849526),
              'mic': (0.15181603033157046, 0.026014072143614259),
              'neu': (-0.18569996860585866, 0.0063177737620680081),
              'oli': (0.14831679103576517, 0.029696726446212698),
              'opc': (0.18120456905503632, 0.0077314488706105283)},
             'gene2': {'ast': (0.077462628897099678, 0.25809556260068212),
              'end': (0.16775942427974594, 0.013778894070374951),
              'mic': (0.03064671931222681, 0.65498013439271485),
              'neu': (-0.086866381704460385, 0.20455325040954411),
              'oli': (-0.076851650607355881, 0.2618805626113867),
              'opc': (0.057551498466516943, 0.40110610344439213)},
             'gene3': {'ast': (0.099641382308193874, 0.14535433530760464),
              'end': (0.1458209567968316, 0.032587664311184109),
              'mic': (0.047519862831751553, 0.48824194134200349),
              'neu': (-0.050456422516844165, 0.46173531503518539),
              'oli': (-0.16359124827935956, 0.016353370642086175),
              'opc': (0.01447632157260499, 0.83285747080654227)},
             'gene4': {'ast': (0.062040860682460335, 0.36532402464482738),
              'end': (0.023928373058996835, 0.72719184233399281),
              'mic': (-0.06301407906493757, 0.35783639254858834),
              'neu': (-0.039194378033760774, 0.56761209151126946),
              'oli': (-0.075391823033640032, 0.27107780500712325),
              'opc': (-0.016166775338694487, 0.81367714402254288)},
             'gene5': {'ast': (0.060541186698543796, 0.37705060491492659),
              'end': (0.055836895360911881, 0.41530560100903868),
              'mic': (-0.060314182907097488, 0.37884551246136011),
              'neu': (-0.062921104107802647, 0.35854754080564777),
              'oli': (-0.051294404598034249, 0.45432035810144566),
              'opc': (-0.0029256924823106087, 0.96598127036589654)},
          'gene6': {'ast': (0.1575937598106692, 0.020790252399645912),
              'end': (0.067940544326112684, 0.32142065367313422),
              'mic': (-0.10574029800284962, 0.12216368533449518),
              'neu': (-0.059433939481755176, 0.38585479753245633),
              'oli': (-0.1016011012098819, 0.13756365746222063),
              'opc': (-0.0016047236108092445, 0.98133701768793424)},
             'gene7': {'ast': (0.08478833103914607, 0.21563287186045033),
              'end': (0.057125262624067238, 0.40460866774480742),
              'mic': (-0.16438938395034897, 0.015830179923102757),
              'neu': (-0.021121012340312492, 0.75814031352908307),
              'oli': (-0.080164939988891293, 0.24180738951749406),
              'opc': (-0.035373952522398514, 0.60597891808772308)},
             'gene8': {'ast': (0.037037842015020889, 0.58912751238797134),
              'end': (0.094246419860416819, 0.16853249020023747),
              'mic': (-0.1291059914511338, 0.058765051269068003),
              'neu': (-0.018021444613489819, 0.79276089120995508),
              'oli': (-0.10036224009273347, 0.14245052518706691),
              'opc': (-0.021582264725059769, 0.75302743889342127)},
             'gene9': {'ast': (0.15678475693689778, 0.021462093046648586),
              'end': (0.12652684199087155, 0.06404278144911503),
              'mic': (-0.056657972904441065, 0.40846938718754444),
              'neu': (-0.15706368180830255, 0.021228368308677966),
              'oli': (-0.078621797193846737, 0.2510184539754034),
              'opc': (0.046871453065758656, 0.49420265751897563)},
             'gene10': {'ast': (0.082579873940447732, 0.22787125199554459),
              'end': (0.053851819652732495, 0.43210785767275173),
              'mic': (-0.13531478664058538, 0.047517154902201854),
              'neu': (-0.034892173199063008, 0.61089793124699765),
              'oli': (-0.095032480861648436, 0.16499370354026677),
              'opc': (-0.026406095293293727, 0.70023788637949658)},
             'gene11': {'ast': (0.2875377333429931, 1.8494132655818492e-05),
              'end': (0.21106401989905577, 0.0018588936809068324),
              'mic': (0.048500326016083457, 0.47930233522356003),
              'neu': (-0.28629524982491733, 2.0168689216522542e-05),
              'oli': (-0.051642155087058363, 0.45126293662022576),
              'opc': (0.15560989156946561, 0.022471154467752021)},}

groups = list(results.keys())
celltypes = ["ast","neu","mic","end","oli","opc"]
translate = {"ast":"Astrocytes",
            "neu":"Neurons",
            "mic":"Microglia",
            "end":"Endothelials",
            "oli":"Oligodendrocytes",
            "opc":"OPCs"}

# Set data
plot_dict = dict()
for cell in celltypes:
    row = []
    for group in groups:
        row.append(results[group][cell][0])
    plot_dict[cell] = row
plot_dict["group"] = groups
df = pd.DataFrame(plot_dict)
 
# ------- PART 1: Create background
 
# number of variable
categories=[translate[i] for i in list(df.columns.drop("group"))]
N = len(categories)
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# If you want the first axis to be on top:
#ax.set_theta_offset(pi / 2)
#ax.set_theta_direction(-1)
 
# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0,0.5,1], ["0","+0.5","+1"], color="grey", size=7)
plt.ylim(-0.4,1.2)
 
# ------- PART 2: Add plots
labels = list()
avgangles = []
avgradii = []
for ix,row in df.iterrows():
    gene = row["group"]
    values=row.drop('group').values.flatten().tolist()
    values += values[:1]
    a, r = average_polar_vec(angles,values,scale=2.5)
    avgangles.append(a)
    avgradii.append(r)
    size = 10
    color = "blue"
    ax.plot(a, r, ms=size,marker='o',c=color)
    if size > 1:
        labels.append(plt.text(a,r,gene,color="k"))
        labeled = True
adjust_text(labels,avgangles,avgradii,arrowprops=dict(arrowstyle='->', color='red'),expand_points=(2,2))

plt.show()
