import numpy as np
import pandas as pd
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

    m = 40 # total 40 groups are there
    #n = len(data_initial)
    M = adj_mtx.shape[0]
    present = np.zeros((M,M))
    '''
    file = open("test_pairs_final","w+")
    for i in range(len(data_initial)):
      x = int(data_initial.iloc[i,0])
      y = int(data_initial.iloc[i,1])
      if present[x,y]==0 and present[y,x]==0:
        present[x,y]=1
        present[y,x]=1
        file.write(str(x)+','+str(y)+'\n')

    file.close()
    print("Original number of pairs=", n)

    data_initial = pd.read_csv('test_pairs_final', sep=',')
    n = len(data_initial)
    print("Old number of pairs=", n)
    total = 0
    #file = open("test_pairs_final","w+")
    file = open("test_pairs_final2", "w+")
    for i in range(len(data_initial)):
      x = int(data_initial.iloc[i,0])
      y = int(data_initial.iloc[i,1])
      if adj_mtx[x,y] > 0:
        file.write(str(x)+','+str(y)+'\n')
        total = total + 1


    for i in range(len(data_initial)):
      x = int(data_initial.iloc[i,0])
      y = int(data_initial.iloc[i,1])
      if adj_mtx[x,y] == 0 and total < 1000001:
        file.write(str(x)+','+str(y)+'\n')
        total = total + 1

    file.close()
    '''
    df1 = pd.read_csv('test_pos_final.txt', sep=',', header=None)
    df2 = pd.read_csv('test_neg_final.txt', sep=',', header=None)
    present = np.zeros((M,M), dtype=int)

    for i in range(len(df2)):
      x = df2.iloc[i,0]
      y = df2.iloc[i,1]
      present[x,y] = 1
      present[y,x] = 1
    for i in range(M):
      present[i,i] = 1
    N = 1000000 - 2*len(df2)
    test_neg_more = []

    for i in range(N):
      x = random.randrange(0,M)
      y = random.randrange(0,M)
      if(present[x,y]==0 and adj_mtx[x,y]==0):
        test_neg_more.append((x,y))
        present[x,y] = 1
        present[y,x] = 1
    df3 = pd.DataFrame(test_neg_more)

    data = pd.concat([df1,df2,df3], ignore_index=True)
    data.to_csv('final_data10.csv')

