import codecs
import numpy as np
import math

print(math.pow(2,784))

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

# fname="../res/model-1-res.txt"
fname="../res/linear model.txt"
fin=codecs.open(fname,'r')
fo=codecs.open("../res/linear model-analysis.txt",'w')

array_index=0
line_index=0
title_line="training steps\tloss\tacc on training_set\tacc on test_set\n"
fo.write(title_line)
fo.flush()

outline=""
# for raw_line in fin.readlines():
#     line_index+=1
#
#     if line_index%4==1:
#         outline=""
#     info=raw_line.strip().split(" ")
#     nl=[]
#     for k in info:
#         if is_number(k):
#             nl.append(k)
#     if line_index % 4 == 1:
#         outline+=nl[0]
#         outline+="\t"
#     if line_index % 4 != 0:
#         outline+=nl[1]
#     if line_index % 4 == 0:
#         outline+="\n"
#         fo.write(outline)
#         fo.flush()
#     else:
#         outline+="\t"


for raw_line in fin.readlines():
    line_index+=1

    if line_index%3==1:
        outline=""
    info=raw_line.strip().split(" ")
    nl=[]
    for k in info:
        if is_number(k):
            nl.append(k)
    if line_index % 3 == 1:
        outline+=nl[0]
        outline+="\t"

    outline+=nl[1]
    if line_index % 3 == 0:
        outline+="\n"
        fo.write(outline)
        fo.flush()
    else:
        outline+="\t"