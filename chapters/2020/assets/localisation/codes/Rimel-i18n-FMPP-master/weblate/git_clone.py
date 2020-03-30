import subprocess
import os

def git(*args):
    return subprocess.check_call(['git'] + list(args))

with open('result_webscrapping.txt', 'r') as file:
    addressL = file.read().replace("{", "").replace("}", "").replace('\'', '').replace(" ","").split(',')


for address in addressL:
	try:
		git("clone", address, "../Projects/"+address[address.rfind('/'):].replace('/',''))
	except:
		print("erreur")
