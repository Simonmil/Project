import numpy as np

numberdict = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,}
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