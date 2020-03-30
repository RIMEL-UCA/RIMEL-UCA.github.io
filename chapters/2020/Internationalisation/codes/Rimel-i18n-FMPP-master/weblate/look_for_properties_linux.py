from path import Path
from collections import defaultdict
import re

pattern = re.compile("^.+_\\w{2}.properties$")
patternJava = re.compile("^.+.java$")

projects = defaultdict(lambda : [0,defaultdict(lambda: 0)])

for f in Path('../Projects/').walkfiles():
	projectName = f.split("/")[2]
	if pattern.match(f):
		projects[projectName][1][f[0:(f.rfind('/'))]] += 1
	if patternJava.match(f):
		projects[projectName][0] +=1


f = open("output/result_properties_top.txt", "w")
for projectname, values in projects.items():
	if len(values[1]):
		text = projectname + "," + str(values[0]) + "="
		for package in values[1]:
			text += package + ':' + str(values[1].get(package))
		f.write(text + ";\n")

f.close()

		


