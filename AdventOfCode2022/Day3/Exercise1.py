import string
import numpy as np

Items = open("Items.txt",'r')

def Priority(Duplicate):
    letters = list(string.ascii_lowercase+string.ascii_uppercase)
    return letters.index(Duplicate) + 1


# Part 1


PrioritySum = 0

for item in Items:
    item = item.strip()
    first = item[:int(len(item)/2)]
    second = item[int(len(item)/2):]

    WrongItem = list(set(first) & set(second))[0]
    PrioritySum += Priority(WrongItem)

print(PrioritySum)

Items.close

# Part 2

Itemsfile = open("Items.txt",'r')
Items = Itemsfile.readlines()

PrioritySum = 0
Group = []

for item in Items:
    Group.append(item.strip())
    if len(Group) == 3:
        Badge = list(set(Group[0]) & set(Group[1]) & set(Group[2]))[0]
        PrioritySum += Priority(Badge)
        Group = []

print(PrioritySum)

Itemsfile.close