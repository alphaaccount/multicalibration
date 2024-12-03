import pandas as pd
import numpy as np
import random

test_pos = pd.read_csv('test_pos.txt')
test_neg = pd.read_csv('test_neg.txt')
train_pos = pd.read_csv('train_pos.txt')
train_neg = pd.read_csv('train_neg.txt')

print(len(test_pos), len(test_neg))
print(len(train_pos), len(train_neg))

### find the nodes which are repeated in both test and train 
### for both positive and negative edges
M = 67796
repeated_pos = np.zeros(M, dtype=int)
repeated_neg = np.zeros(M, dtype=int)

# 1 means present in test and 2 means present in both train and test

for i in range(len(test_pos)):
  x = int(test_pos.iloc[i,0])
  y = int(test_pos.iloc[i,1])
  repeated_pos[x] = 1
  repeated_pos[y] = 1

for i in range(len(train_pos)):
  x = int(train_pos.iloc[i,0])
  y = int(train_pos.iloc[i,1])
  if repeated_pos[x] == 1:
    repeated_pos[x] = 2
  if repeated_pos[y] == 1:
    repeated_pos[y] = 2

test_pos_list = []
train_pos_list = []
marked = np.zeros(M, dtype=int)
deleted_from_test = np.zeros(M, dtype=int)
repeated = np.count_nonzero(repeated_pos==2)
print(repeated)
from_test = int(0.1*repeated)
from_train = int(0.9*repeated)
test_pos1 = test_pos
counter = 0

for i in range(len(test_pos1)):
  x = int(test_pos1.iloc[i,0])
  y = int(test_pos1.iloc[i,1])
  if repeated_pos[x]==2 and counter <= from_test and marked[x]==0:
    counter = counter+1
    marked[x] = 1
    deleted_from_test[x] = 1
  if repeated_pos[y]==2 and counter <= from_test and marked[y]==0:
    counter = counter+1
    marked[y] = 1
    deleted_from_test[y] = 1
  if marked[x]==0 and marked[y]==0:
    test_pos_list.append((x,y))

print(len(test_pos_list))
pd.DataFrame(test_pos_list).to_csv('test_pos_final_less.txt', header=False, sep=',', index=False)

counter = 0
train_pos_list = []
marked1 = np.zeros(M, dtype=int)
for i in range(len(train_pos)):
  x = int(train_pos.iloc[i,0])
  y = int(train_pos.iloc[i,1])
  #if repeated_pos[x]==2 and repeated_pos[y]==2:
  #  train_pos_list.append((x,y))
  if repeated_pos[x]==2 and counter <= from_train and marked[x]==0:
    counter = counter+1
    marked[x] = 1
  if repeated_pos[y]==2 and counter <= from_train and marked[y]==0:
    counter = counter+1
    marked[y] = 1
  if marked[x]==0 and marked[y]==0:
    train_pos_list.append((x,y))

print(len(train_pos_list))
pd.DataFrame(train_pos_list).to_csv('train_pos_final_less.txt', sep=',', header=False, index=False )



repeated_neg = np.zeros(M, dtype=int)
for i in range(len(test_neg)):
  x = int(test_neg.iloc[i,0])
  y = int(test_neg.iloc[i,1])
  repeated_neg[x] = 1
  repeated_neg[y] = 1

for i in range(len(train_neg)):
  x = int(train_neg.iloc[i,0])
  y = int(train_neg.iloc[i,1])
  if repeated_neg[x] == 1:
    repeated_neg[x] = 2
  if repeated_neg[y] == 1:
    repeated_neg[y] = 2

test_neg_list = []
train_neg_list = []
marked1 = np.zeros(M, dtype=int)
deleted_from_test = np.zeros(M, dtype=int)
repeated = np.count_nonzero(repeated_neg==2)
print(repeated)
from_test = int(0.1*repeated)
from_train = int(0.9*repeated)

counter = 0

N1 = len(test_pos_list)
N2 = len(train_pos_list)
N11 = len(test_neg)
N22 = len(train_neg)
S1 = random.sample(range(N11), N1)
S2 = random.sample(range(N22), N2)
#print(N1, N11)
#print(N2, N22)
#print(S1)
#print(S2)
for i in S1:
  x = int(test_neg.iloc[i,0])
  y = int(test_neg.iloc[i,1])
  #print(x,y)
  test_neg_list.append((x,y))

for i in S2:
  x = int(train_neg.iloc[i,0])
  y = int(train_neg.iloc[i,1])
  train_neg_list.append((x,y))


print(len(test_neg_list))
pd.DataFrame(test_neg_list).to_csv('test_neg_final_less.txt', sep=',', header=False, index=False)


print(len(train_neg_list))
pd.DataFrame(train_neg_list).to_csv('train_neg_final_less.txt', sep=',', header=False, index=False)
