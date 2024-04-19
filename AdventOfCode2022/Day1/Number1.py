import numpy as np

file  = open("CaloriesList.txt")
Elves = []
Cals = 0

for line in file:
    try:
        Cals += int(line)
    except:
        Elves.append(Cals)
        Cals = 0

print(max(Elves))

# Part 2

ElvesSorted = sorted(Elves,reverse=True)
#print(ElvesSorted)

TopThreeCalsSum = sum(ElvesSorted[0:3])
print(TopThreeCalsSum)