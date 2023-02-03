a=3
b=4
a==b
a>b
a<b
3<4
print('shweta' == 'shweta')
x="vishwa"
y=len(x)
y
if (y>5):
    print("y is grater than 5")
elif(y<5):
    print("y is less than 5")
else:
    print("y is equal to 5")
vaccine=input()
num_doses=int(input())

# clean string using string manupulation
vaccine = vaccine.replace(' ', '') # replace extra spaces
vaccine = vaccine.replace('co-vaxin', 'covaxin') # replace '-'  with nothing
vaccine = vaccine.lower() # lower case all the charactersa in string


if vaccine=='covishield' and num_doses==2:
    print('Fully vaccinated')
elif vaccine=='covaxin' and num_doses==2:
    print('Fully vaccinated')
elif vaccine=='sputnik' and num_doses==1:
    print('Fully vaccinated')
else:
    print("Not fully vaccinated")
x=7
if (x>5):
    x*2
    print("x is greater than 5")
else:
    print("x is less than 5")
    

