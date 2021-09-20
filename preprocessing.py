
import numpy as np
import pandas as pd
import datetime

# Path to Original Training Dataset "Clicks" File
dataBefore = 'C:/Users/Kinjalk Parth/Desktop/Recommeder System/Practical/GRUSessionBased/GRU4REC-pytorch-master/data/raw_data/yoochoose-clicks.dat'
#Path to Original Testing Dataset "Clicks" File
dataTestBefore = 'C:/Users/Kinjalk Parth/Desktop/Recommeder System/Practical/GRUSessionBased/GRU4REC-pytorch-master/data/raw_data/yoochoose-test.dat'

# Path to Processed Dataset Folder
dataAfter = 'C:/Users/Kinjalk Parth/Desktop/Recommeder System/Practical/STAMP_analysis'

dayTime = 86400 #Validation Only one day = 86400 seconds

def removeShortSessions(data):
    #delete sessions of length =< 4
    sessionLen = data.groupby('SessionID').size() #group by sessionID and get size of each session
    data = data[np.isin(data.SessionID, sessionLen[sessionLen > 4].index)]
    return data

#Read Dataset in pandas Dataframe (Ignore Category Column)
train = pd.read_csv(dataBefore, sep=',', header=None, usecols=[0,1,2], dtype={0:np.int32, 1:str, 2:np.int64})
test = pd.read_csv(dataTestBefore, sep=',', header=None, usecols=[0,1,2], dtype={0:np.int32, 1:str, 2:np.int64})
train.columns = ['SessionID', 'Time', 'ItemID'] #Headers of dataframe
test.columns = ['SessionID', 'Time', 'ItemID'] #Headers of dataframe
train['Time']= train.Time.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()) #Convert time objects to timestamp
test['Time'] = test.Time.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()) #Convert time objects to timestamp

#remove sessions of less than 2 interactions
train = removeShortSessions(train)
#delete records of items which appeared less than 5 times
itemLen = train.groupby('ItemID').size() #groupby itemID and get size of each item
train = train[np.in1d(train.ItemID, itemLen[itemLen > 4].index)]
#remove sessions of less than 2 interactions again
train = removeShortSessions(train)

######################################################################################################3

#Delete records in testing split where items are not in training split
test = test[np.in1d(test.ItemID, train.ItemID)]
#Delete Sessions in testing split which are less than 2
test = removeShortSessions(test)

#Convert To CSV
#print('Full Training Set has', len(train), 'Events, ', train.SessionID.nunique(), 'Sessions, and', train.ItemID.nunique(), 'Items\n\n')
#train.to_csv(dataAfter + 'recSys15TrainFull.txt', sep='\t', index=False)
print('Testing Set has', len(test), 'Events, ', test.SessionID.nunique(), 'Sessions, and', test.ItemID.nunique(), 'Items\n\n')
test.to_csv(dataAfter + 'STAMP_DATA_TEST.txt', sep=',', index=False)

######################################################################################################3
#Separate Training set into Train and Validation Splits
timeMax = train.Time.max()
sessionMaxTime = train.groupby('SessionID').Time.max()
sessionTrain = sessionMaxTime[sessionMaxTime < (timeMax - dayTime)].index #training split is all sessions that ended before the last 2nd day
sessionValid = sessionMaxTime[sessionMaxTime >= (timeMax - dayTime)].index #validation split is all sessions that ended during the last 2nd day
trainTR = train[np.in1d(train.SessionID, sessionTrain)]
trainVD = train[np.in1d(train.SessionID, sessionValid)]
#Delete records in validation split where items are not in training split
trainVD = trainVD[np.in1d(trainVD.ItemID, trainTR.ItemID)]
#Delete Sessions in testing split which are less than 2
trainVD = removeShortSessions(trainVD)
#Convert To CSV
print('Training Set has', len(trainTR), 'Events, ', trainTR.SessionID.nunique(), 'Sessions, and', trainTR.ItemID.nunique(), 'Items\n\n')
trainTR.to_csv(dataAfter + 'STAMP_DATA_TRAIN.txt', sep=',', index=False)
print('Validation Set has', len(trainVD), 'Events, ', trainVD.SessionID.nunique(), 'Sessions, and', trainVD.ItemID.nunique(), 'Items\n\n')
trainVD.to_csv(dataAfter + 'STAMP_DATA_VALID.txt', sep=',', index=False)
