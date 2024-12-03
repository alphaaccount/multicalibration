import pandas as pd
import numpy as np
import random
M = 67796
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


for j in range(10):
    df1 = pd.read_csv('test_pos_final2.txt', sep=',', header=None)
    df2 = pd.read_csv('test_neg_final2.txt', sep=',', header=None)
    present = np.zeros((M,M), dtype=int)
    print(len(df1))
    print(len(df2))
    for i in range(len(df2)):
      x = df2.iloc[i,0]
      y = df2.iloc[i,1]
      present[x,y] = 1
      present[y,x] = 1
    for i in range(M):
      present[i,i] = 1
    N = 1000000 - 2*len(df2)
    test_neg_more = []

    Y = pd.read_csv("../../../../UGE-Unbiased-Graph-Embedding/processed_data/deleted_nodes.csv")
    deleted_nodes = set(Y["Index"])
    for i in range(N):
      x = random.randrange(0,M)
      y = random.randrange(0,M)
      if(present[x,y]==0 and adj_mtx[x,y]==0 and (x not in deleted_nodes) and (y not in deleted_nodes)):
        test_neg_more.append((x,y))
        present[x,y] = 1
        present[y,x] = 1
    df3 = pd.DataFrame(test_neg_more)
    print(len(df3))
    data = pd.concat([df1,df2,df3], ignore_index=True)
    #####   DO SOME PREPROESSING HERE SO THAT RANDOM K PAIRS ARE AT THE TOP OF THE DATA ##############
    index_list = []
    non_index_list = []
    index = random.sample(range(len(data)), 10000)
    for i in range(len(data)):
      x = int(data.iloc[i,0])
      y = int(data.iloc[i,1])
      if i in index:
        index_list.append([x,y])
      else:
        non_index_list.append([x,y])
    dataf1 = pd.DataFrame(index_list)
    dataf2 = pd.DataFrame(non_index_list)
    data = pd.concat([dataf1, dataf2], ignore_index=True)
    name = 'data_new'+str(j)+'.csv'
    data.to_csv(name, header=False, index=False)
    print('done',j)


