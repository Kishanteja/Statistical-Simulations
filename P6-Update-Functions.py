import math

def UpdateMean (OldMean , NewDataValue , n , A) :
    NewMean = (OldMean*n + NewDataValue)/(n+1)
    return NewMean

def UpdateMedian (OldMedian , NewDataValue , n , A) :
    A.append(NewDataValue)
    A.sort()
    if n//2 == 0 :
        NewMedian = ( A[n//2] + A[n//2 + 1] )/ 2.0
    else :
        NewMedian = A[n//2 + 1]
    return NewMedian

def UpdateStd ( OldMean , OldStd , NewMean , NewDataValue , n , A) :
    OldVar = OldStd**2
    NewVar = ( n*( OldVar + OldMean**2 ) - (NewMean**2)(n+1) + (NewDataValue)**2 ) / (n+1) 
    NewStd = math.sqrt(NewVar)
    return NewStd

