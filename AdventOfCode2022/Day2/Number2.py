#import numpy

# A and X is Rock, B and Y is paper, C and Z is scissors
ResultPoints = {'Defeat':0,'Tie':3,'Victory':6}
ShapePoints = {'A':1,'B':2,'C':3,'X':1,'Y':2,'Z':3}
ResultCodes = {'X':'Defeat','Y':'Tie','Z':'Victory'}

Guide = open("StrategyGuide.txt")

def check(diff):
    if abs(diff) == 2:
        if diff < 0:
            result = "Defeat"
        elif diff > 0:
            result = "Victory"
    elif abs(diff) == 1:
        if diff > 0:
            result = "Defeat"
        elif diff < 0:
            result = "Victory"
    else:
        result = "Tie"
    
    return result

def ChooseShape(Code):
    if Code == 'X':
        match Opponent:
            case 'A':
                me = 'C'
            case 'B':
                me = 'A'
            case 'C':
                me = 'B'
    elif Code == 'Z':
        match Opponent:
            case 'A':
                me = 'B'
            case 'B':
                me = 'C'
            case 'C':
                me = 'A'
    else:
        me = Opponent
    return me

# Part 1

TotalPoints = 0

for line in Guide:
    Opponent = line.split()[0]
    me = line.split()[1]
    #print("Opponent:",Opponent,"Me:",me)
    
    
    diff = ShapePoints[Opponent] - ShapePoints[me]
    result = check(diff)
    TotalPoints += ShapePoints[me] + ResultPoints[result]

print(TotalPoints)

Guide.close

# Part 2
# X is Lose, Y is Draw, and Z is Win

Guide = open("StrategyGuide.txt")
TotalPoints = 0

for line in Guide:
    Opponent = line.split()[0]
    Code = line.split()[1]

    me = ChooseShape(Code)
    TotalPoints += ShapePoints[me] + ResultPoints[ResultCodes[Code]]
    #print(Code, Opponent, me)

print(TotalPoints)
