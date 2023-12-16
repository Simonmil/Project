import numpy as np

file = open("Input.txt","r")

lines = file.readlines()
sum = 0
for line in lines:
    values = []
    
    
    for j in range(len(line)):
        try: 
            value = int(line[j])
            values.append(str(value))
        except:
            continue
    if len(values) == 1:
        number = values[0] + values[0]
    else:
        number = values[0] + values[-1]
    sum += int(number)
    
    
file.close
print(sum)