import numpy as np


file = open("Procedure.txt",'r')

procedures = file.readlines()

for line in procedures:
    print(line)