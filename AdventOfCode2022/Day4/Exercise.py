import numpy as np

file = open("sections.txt",'r')
Sections = file.readlines()
file.close()
# Part 1

Contains = 0

for pair in Sections:
    assignments = pair.strip()
    first = assignments.split(",")[0]
    last = assignments.split(",")[1]

    if int(first.split("-")[0]) <= int(last.split("-")[0]) and int(first.split("-")[1]) >= int(last.split("-")[1]):
        Contains += 1
    elif int(first.split("-")[0]) >= int(last.split("-")[0]) and int(first.split("-")[1]) <= int(last.split("-")[1]):
        Contains += 1


print(Contains)

# Part 2

Overlap = 0


for pair in Sections:
    assignments = pair.strip()
    first = [assignments.split(",")[0].split("-")[0],assignments.split(",")[0].split("-")[1]]
    last = [assignments.split(",")[1].split("-")[0],assignments.split(",")[1].split("-")[1]]

    if int(last[0]) < int(first[0]) and int(last[1]) > int(first[1]):
        interval = list(range(int(last[0]),int(last[1])+1))
    else:
        interval = list(range(int(first[0]),int(first[1])+1))
    #print(interval,first,last)
    if int(last[0]) in interval or int(last[1]) in interval:
        Overlap += 1

print(Overlap)
    