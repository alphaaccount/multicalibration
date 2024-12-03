import numpy as np
import pandas as pd
import networkx as nx
import random
M = 67796
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
adj_mtx = np.zeros((M,M), dtype=int)
# load the adjacency matrix of size MxM of training or testing set

print('loading data ...')
with open('pokec-z_edge.csv', 'r') as fin:
    lines = fin.readlines()
    idx = 0
    for line in lines:
        if idx == 0:
            idx += 1
            continue
        eachline = line.strip().split(',')
        scr_node = int(eachline[0])
        dst_node = int(eachline[1])
        weight = float(eachline[2])
        adj_mtx[scr_node, dst_node] = weight
        adj_mtx[dst_node, scr_node] = weight
        idx += 1


#again repeating what I said first I will do a 90-10 split
#then do that 90 10 grouping thing then problem is negative 
#edges will be much more, then replicate 

edge_num = np.count_nonzero(adj_mtx)*0.5
print("Number of edges are", edge_num)
neg_edge_num = (M*M - M - edge_num)*0.5
print("Number of negative edges are", neg_edge_num)

#### Don't replicate edges ###############

test_count_pos = int(0.1*edge_num)
train_count_pos = int(0.9*edge_num)

test_count_neg = test_count_pos
train_count_neg = train_count_pos

counter_pos = 0
counter_neg = 0

f_test_pos = open('test_pos.txt','w+')
f_test_neg = open('test_neg.txt','w+')
f_train_pos = open('train_pos.txt','w+')
f_train_neg = open('train_neg.txt', 'w+')

present = np.zeros((M,M), dtype=int)

for i in range(M):
    present[i,i]=1 


with open('pokec-z_edge1.csv', 'r') as fin:
    lines = fin.readlines()
    #print(lines)
    random.shuffle(lines)
    #print(lines)
    idx = 0
    for line in lines:
        eachline = line.strip().split(',')
        #print(eachline[0], eachline[1])
        i = int(eachline[0])
        j = int(eachline[1])
        idx += 1
        if present[i,j]==0 and counter_pos <= test_count_pos:
          f_test_pos.write(str(i)+','+str(j)+'\n')
          counter_pos = counter_pos + 1
          present[i,j] = 1
          present[j,i] = 1
        elif present[i,j]==0  and counter_pos > test_count_pos and counter_pos <= train_count_pos:
          f_train_pos.write(str(i)+','+str(j)+'\n')
          counter_pos = counter_pos + 1
          present[i,j] = 1
          present[j,i] = 1

print("######Positive Data written to File#########")
'''
G=nx.from_numpy_matrix(adj_mtx)
G1 = nx.complement(G)
l = G1.edges()
N = len(l)
print(N)
S = random.sample(range(N), test_count_neg)
for i in S:
  (x,y) = l[i]
  f_test_neg.write(str(x)+','+str(y)+'\n')
  l[:i] + l[i+1:]


N1 = len(l)
print(N1)
S = random.sample(range(N1), train_count_neg)
for i in S:
  (x,y) = l[i]
  f_train_neg.write(str(x)+','+str(y)+'\n')


X = M
Y = M
x = y = 0
dx = 0
dy = -1
counter_neg = 0
for i in range(max(X, Y)**2):
    if (-X/2 < x <= X/2) and (-Y/2 < y <= Y/2):
        i = int(x + M/2)
        j = int(y + M/2)
        if adj_mtx[i,j]==0 and counter_neg <= test_count_neg:
          f_test_neg.write(str(i)+','+str(j)+'\n')
          counter_neg = counter_neg + 1
        if adj_mtx[i,j]==0 and counter_neg > test_count_neg and counter_neg <= train_count_neg :
          f_train_neg.write(str(i)+','+str(j)+'\n')
          counter_neg = counter_neg + 1
        if counter_neg > train_count_neg:
          break
    if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
        dx, dy = -dy, dx
    x, y = x+dx, y+dy
'''
visited = np.zeros((M,M), dtype=int)

for x in range(M):
  for y in range(M):
    i = random.randrange(M)
    j = random.randrange(M)
    if visited[i,j]==0 and adj_mtx[i,j]==0 and counter_neg <= test_count_neg:
      f_test_neg.write(str(i)+','+str(j)+'\n')
      counter_neg = counter_neg + 1
      visited[i,j]=1
      visited[j,i]=1
    if visited[i,j]==0 and adj_mtx[i,j]==0 and counter_neg > test_count_neg and counter_neg <= train_count_neg :
      f_train_neg.write(str(i)+','+str(j)+'\n')
      counter_neg = counter_neg + 1
      visited[i,j]=1
      visited[j,i]=1
    if counter_neg >= train_count_neg:
      break



print("####Negative data written on File")


'''
for i in range(M):
    for j in range(M):
      if present[i,j]==0 and adj_mtx[i,j]==0 and counter_neg <= test_count_neg:
        f_test_neg.write(str(i)+','+str(j)+'\n')
        counter_neg = counter_neg + 1
        present[i,j] = 1
        present[j,i] = 1
      elif present[i,j]==0 and adj_mtx[i,j]==0 and counter_neg > test_count_neg and counter_neg <= train_count_neg :
        f_train_neg.write(str(i)+','+str(j)+'\n')
        counter_neg = counter_neg + 1
        present[i,j] = 1
        present[j,i] = 1
      if counter_neg > train_count_neg:
        break

print("####Train Data written to File#######")


    elif present[i,j]==0 and adj_mtx[i,j]==0 and counter_test_neg <= test_count_neg:
      f_test_neg.write(str(i)+','+str(j))
      counter_test_neg = counter_test_neg + 1
      present[i,j] = 1
      present[j,i] = 1
    elif present[i,j]==0 and adj_mtx[i,j]==0 and counter_test_neg > test_count_neg:
      f_train_neg.write(str(i)+','+str(j))
'''

test_pos = pd.read_csv('test_pos.txt')
test_neg = pd.read_csv('test_neg.txt')
train_pos = pd.read_csv('train_pos.txt')
train_neg = pd.read_csv('train_neg.txt')

print(len(test_pos), len(test_neg))
print(len(train_pos), len(train_neg))

