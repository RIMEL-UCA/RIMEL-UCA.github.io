from path import Path
from collections import defaultdict
import re

pattern = re.compile("^.+values(-[a-zA-Z]{1,3})+(\\+\\w{2,5})*\\\\.+$")
patternJava = re.compile("^.+.java$")
projects = defaultdict(lambda : [0,defaultdict(lambda: 0)])


valueFolders = {};

for f in Path('../Projects').walkfiles():
	projectName = f.split("\\")[1]
	if patternJava.match(f):
		projects[projectName][0] +=1


	if pattern.match(f):
		valueFolder = f[0:(f.rfind('\\'))]
		resFolder = valueFolder[0:(valueFolder.rfind('\\'))]
		if valueFolder not in valueFolders:
			valueFolders[valueFolder] = 1
			projects[projectName][1][resFolder] += 1
			
		
f = open("output/result_properties_android.txt", "w")
for projectname, values in projects.items():
	if len(values[1]):
		text = projectname + "," + str(values[0]) + "="
		for resFolder in values[1]:
			text += resFolder + ':' + str(values[1].get(resFolder))
		f.write(text + ";\n")

f.close()
		


