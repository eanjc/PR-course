import numpy as np
import codecs

t1=0
t2=0

text1=""
text2=""
while t1<20 or t2<20:
    t=np.random.ranf()
    if t>0.5:
        f1=1
    else:
        f1=-1
    t = np.random.ranf()
    if t>0.5:
        f2=1
    else:
        f2=-1
    a=np.random.ranf()*8*f1
    b=np.random.ranf()*8*f2
    if a*a+b*b<4:
        text1+=str(a)+"\t "+str(b)+"\n"
        t1+=1
    else:
        text2 += str(a) + "\t " + str(b)+"\n"
        t2+=1

fo1=codecs.open("../res/num1.txt",'w','utf-8')
fo1.write(text1)
fo2=codecs.open("../res/num2.txt",'w','utf-8')
fo2.write(text2)